"""Episode reset and scene-derived AutoOpen state initialization."""

import math

import numpy as np

from gr00t.rl.auto_open.control import (
    APPROACH_ARRIVAL,
    ROBOT_TRUNK_REAR_X,
    THROUGH_TARGET_LEAD_DISTANCE,
    THROUGH_TAIL_TARGET_DISTANCE,
    _effective_grasp_target,
    _quat_distance,
)
from gr00t.rl.auto_open.geometry import (
    _initialize_gripper_limits,
    _initialize_panel_push_tcp,
    _introspect_robot_colliders,
    _matrix_to_quat_np,
    _measure_handle_geometry,
    _measure_panel_geometry,
    _physical_handle_grasp_position,
    _quat_conjugate_np,
    _quat_rotate_inverse_np,
    _quat_rotate_np,
    _quat_rotate_points_np,
)
from gr00t.rl.auto_open.utils import as_numpy as _as_np

def reset_controller(self, state=None, reset_env=True, close_gripper=True):
    if reset_env:
        self.env.reset()
        self.runner.obs = self.env.get_observations()
    self.stage = type(self.stage).APPROACH
    self.stage_steps = 0
    self.base_hold_stage = None
    self.base_hold_target_xy = None
    self.base_hold_target_yaw = None
    self.base_hold_nominal_xy = None
    self.base_hold_nominal_yaw = None
    self.base_command_previous = np.zeros(3, dtype=np.float32)
    self.base_reversal_pending_sign = np.zeros(3, dtype=np.int8)
    self.base_reversal_pending_steps = np.zeros(3, dtype=np.int32)
    self.base_reversal_lock_steps = np.zeros(3, dtype=np.int32)
    self.base_arrival_settle_steps = 0
    self.base_arrival_confirmed = False
    self.base_arrival_profile = APPROACH_ARRIVAL
    self.pregrasp_settle_steps = 0
    self.handle_align_settle_steps = 0
    self.grasp_settle_steps = 0
    self.handle_contact_last_seen[:] = -1000
    self.handle_contact_last_points[:] = np.nan
    self.grasp_bilateral_contact_seen = False
    self.grasp_waypoint_world = None
    self.center_insert_waypoint_world = None
    self.grasp_branch = None
    self.turn_start_handle_angle = 0.0
    self.turn_start_latch_position = 0.0
    self.turn_start_hinge_angle = 0.0
    self.turn_last_handle_angle = 0.0
    self.turn_last_hinge_angle = 0.0
    self.turn_handle_axis_world = None
    self.turn_axle_world = None
    self.turn_grasp_offset_world = None
    self.turn_grasp_orientation = None
    self.turn_target_handle_angle = None
    self.turn_target_hinge_angle = None
    self.turn_endpoint_world = None
    self.turn_endpoint_quat = None
    self.turn_path_world = None
    self.turn_path_quat = None
    self.turn_path_index = 0
    self.turn_path_stage = None
    self.turn_path_replan_count = 0
    self.turn_gap_started = False
    self.turn_release_slow_steps = 0
    self.turn_command_bias_world[:] = 0.0
    self.release_start_world = None
    self.release_start_quat = None
    self.release_endpoint_world = None
    self.release_endpoint_quat = None
    self.release_start_handle_angle = 0.0
    self.release_start_hinge_angle = 0.0
    self.release_return_axle_world = None
    self.release_return_axis_world = None
    self.release_return_offset_world = None
    self.release_return_start_quat = None
    self.release_vertical_delta = 0.0
    self.release_stable_steps = 0
    self.release_open_start_world = None
    self.release_open_command_path = None
    self.release_open_orientation_path = None
    self.release_retreat_start_world = None
    self.release_retreat_target_world = None
    self.release_retreat_command_target_world = None
    self.release_retreat_command_path = None
    self.release_retreat_orientation_path = None
    self.release_retreat_path_index = 0
    self.release_restore_command_path = None
    self.release_restore_orientation_path = None
    self.arm_retract_entry_count = 0
    self.arm_retract_randomization_records = []
    self.active_arm_retract_pose_local = self.arm_retract_pose_local.copy()
    self.restore_retract_pose_sampled = False
    self.panel_contact_target_world = None
    self.panel_contact_desired_ee_world = None
    self.panel_contact_command_target_world = None
    self.panel_contact_inverse_target_world = None
    self.panel_contact_inverse_orientation_world = None
    self.panel_contact_inverse_command_world = None
    self.panel_contact_inverse_command_orientation_world = None
    self.prepared_panel_contact_command = None
    self.panel_contact_selected_offset = None
    self.batched_panel_finger_clearance = None
    self.defer_path_solve = False
    self.panel_contact_orientation = None
    self.panel_contact_settle_steps = 0
    self.panel_align_ee_goal_local = None
    self.panel_align_ee_orientation_local = None
    self.align_panel_arm_joint_target = None
    self.base_advance_arm_joint_target = None
    self.push_start_hinge_angle = None
    self.push_last_progress_hinge = None
    self.push_no_progress_steps = 0
    self.push_pause_steps = 0
    self.body_align_target_xy = None
    self.body_align_settle_steps = 0
    self.push_contact_world = None
    self.push_normal_world = None
    self.push_orientation = None
    self.push_peak_hinge = 0.0
    self.push_coast_steps = 0
    self.push_depth = 0.0
    self.push_command_hinge_angle = None
    self.push_max_observed_angular_speed = 0.0
    self.push_speed_error_integral = 0.0
    self.push_finger_clearance = float("nan")
    self.push_finger_approach_speed = float("nan")
    self.push_target_finger_clearance = float("nan")
    self.push_finger_force = 0.0
    self.push_contact_retreat_active = False
    self.push_previous_finger_clearance = None
    self.push_arm_detached = False
    self.push_detached_ee_goal_local = None
    self.push_recovery_settle_steps = 0
    self.push_comfort_reach_radius = None
    self.push_reengage_reachable = False
    self.push_overspeed_steps = 0
    self.push_base_start_xy = None
    self.push_panel_contact_start_world = None
    self.base_advance_target_longitudinal = None
    self.push_shoulder_min_clearance = float("nan")
    self.push_shoulder_predicted_min_clearance = float("nan")
    self.push_base_safety_scale = 1.0
    self.through_best_station_error = math.inf
    self.through_no_progress_steps = 0
    self.through_settle_steps = 0
    self.through_stop_latched = False
    self.through_stability_failure_counts = {}
    self.through_motion_ready = False
    self.through_entry_stable_steps = 0
    self.push_target_hinge_angle = None
    self.failure_reason = None
    self.failure_stage = None
    self.last_acceptance_diagnostics = {}
    self.last_stability_diagnostics = {}
    self.stage_exit_diagnostics = {}
    self.precomputed_steady_paths = {}
    self.center_orientation_settle_steps = 0
    self.pregrasp_clearance_fallback = False
    self.center_clearance_fallback = False
    self.center_vertical_bias = 0.02
    self.center_depth_bias = -0.025
    self.gripper_open_settle = 0
    self.pregrasp_pitch_deg = 0.0
    self.center_tangent_feedback_world[:] = 0.0
    self.center_normal_feedback = 0.0
    self.arm_align_command_world = None
    self.arm_align_command_quat = None
    self.arm_align_command_path = None
    self.arm_align_orientation_path = None
    self.arm_align_path_index = 0
    self.arm_forward_target_world = None
    self.arm_forward_desired_target_world = None
    self.arm_align_desired_target_world = None
    self.arm_forward_command_world = None
    self.arm_forward_command_quat = None
    self.arm_forward_command_path = None
    self.arm_forward_orientation_path = None
    self.arm_forward_path_index = 0
    self.center_vertical_plan = None
    self.arm_stability_window.clear()
    self.grasp_latched = False
    self.grasp_vertical_lift = 0.0
    self.grasp_vertical_lift_target = 0.0
    self.grasp_vertical_balance_started = False
    if state is None:
        state = self.runner.get_control_state()
    door_root = _as_np(state["door_root_pose_w"])[0]
    door_bodies = _as_np(state["door_body_state_w"])[0]
    handle_state = door_bodies[self.handle_body_idx]
    grasp_state = (
        door_bodies[self.grasp_target_body_idx]
        if self.grasp_target_body_idx is not None
        else handle_state
    )
    panel_state = door_bodies[self.panel_body_idx]
    base = _as_np(state["root_states"])[0]
    # The generated asset is an X-thin door spanning Y.  Keep the side
    # convention explicit so all four directions use the same geometry.
    side_sign = 1.0 if self.robot_side == "positive_x" else -1.0
    self.door_normal = np.array([side_sign, 0.0, 0.0], dtype=np.float32)
    normal = self.door_normal[:2]
    # Fail early if reset/configuration put the robot on the wrong side.
    side_error = float((base[0] - door_root[0]) * side_sign)
    if side_error < 0.25:
        raise RuntimeError(
            f"robot is not on requested {self.robot_side} side: "
            f"base_x={base[0]:.3f}, door_x={door_root[0]:.3f}"
        )
    self.handle_pos = handle_state[:3]
    self.handle_collider_center_local, handle_radius, self.handle_collider_path = (
        _measure_handle_geometry(self, handle_state, grasp_state)
    )
    _measure_panel_geometry(self, )
    if self.panel_center_world is None or self.hinge_world is None:
        raise RuntimeError("door panel or hinge geometry is unavailable")
    self.grasp_center_pos = _physical_handle_grasp_position(self, handle_state)
    self.handle_free_axis_world = _quat_rotate_np(
        handle_state[3:7], self.handle_free_axis_local
    )
    self.handle_free_axis_world /= max(
        float(np.linalg.norm(self.handle_free_axis_world)), 1.0e-8
    )
    # Define the push point in door-local physical coordinates rather than
    # as a fraction of the randomized panel dimensions.
    self.panel_push_world = self.hinge_world.copy()
    self.panel_push_world += (
        self.panel_push_tangent_offset
        * self.panel_hinge_to_free_direction
    )
    self.panel_push_world[2] = (
        self.panel_bottom_z + self.panel_push_height_offset
    )
    panel_midplane_projection = float(
        np.dot(self.panel_center_world, self.door_normal)
    )
    self.panel_push_world += (
        panel_midplane_projection
        - float(np.dot(self.panel_push_world, self.door_normal))
    ) * self.door_normal
    door_tangent = np.array(
        [-self.door_normal[1], self.door_normal[0]], dtype=np.float32
    )
    panel_contact_world = self.panel_push_world + self.door_normal * self.panel_half_thickness
    self.panel_contact_local = _quat_rotate_inverse_np(
        panel_state[3:7], panel_contact_world - panel_state[:3]
    )
    self.handle_lower_z = float(self.grasp_center_pos[2] - handle_radius)
    self.handle_upper_z = float(self.grasp_center_pos[2] + handle_radius)
    # ``normal`` points from the door toward the robot's initial side, so
    # the pre-door station lies along +normal from the handle.
    # The locomotion policy has a repeatable stopping offset near the
    # frame.  Keep the
    # geometric target here for logging; arrival tolerances are owned by
    # the shared base-target tracker.
    # Keep the refined 0.20 m fixed jaw outside the handle while the arm
    # remains in its randomized middle preset.  At 0.65 m the moving jaw
    # reaches the lever during locomotion and invalidates the precomputed
    # closed-door trajectory; 0.77 m retains arm reach with 0.12 m more
    # clearance.
    self.approach_distance = 0.77
    self.station_xy = self.grasp_center_pos[:2] + self.approach_distance * normal
    self.panel_station_xy = self.station_xy.copy()
    self.panel_station_xy += float(
        np.dot(self.panel_center_world[:2] - self.panel_station_xy, door_tangent)
    ) * door_tangent
    self.door_forward = -self.door_normal
    self.through_station_yaw = math.atan2(
        float(self.door_forward[1]), float(self.door_forward[0])
    )
    c = math.cos(self.through_station_yaw)
    s = math.sin(self.through_station_yaw)
    rear_offset_world = np.array(
        [c * ROBOT_TRUNK_REAR_X, s * ROBOT_TRUNK_REAR_X],
        dtype=np.float32,
    )
    target_tail_xy = (
        door_root[:2]
        + (THROUGH_TAIL_TARGET_DISTANCE + THROUGH_TARGET_LEAD_DISTANCE)
        * self.door_forward[:2]
    )
    self.through_station_xy = target_tail_xy - rear_offset_world
    self.initial_handle_pos = self.grasp_center_pos.copy()
    self.initial_door_root = door_root.copy()
    self.default_ee_goal_local = _as_np(state["ee_goal_cart_local"])[0]
    self.default_ee_goal_delta_rpy = _as_np(state["ee_goal_delta_rpy"])[0]
    middle_arm_joint_target = np.asarray(
        self.env.cfg.doorman_arm_joint_init_center, dtype=np.float32
    )
    if middle_arm_joint_target.shape != (len(self.env.arm_joint_names),):
        raise ValueError(
            "DoorMan middle arm preset must contain one value per arm joint: "
            f"expected {len(self.env.arm_joint_names)}, "
            f"got {middle_arm_joint_target.shape}"
        )
    self.middle_arm_joint_target = middle_arm_joint_target
    self.ee_command_world = _as_np(state["ee_pos"])[0].copy()
    self.initial_ee_orientation = _as_np(state["ee_orn"])[0].copy()
    dof_names = list(state["dof_names"])
    dof_position = _as_np(state["dof_pos"])[0]
    self.initial_arm_joint_target = np.asarray(
        [dof_position[dof_names.index(name)] for name in self.env.arm_joint_names],
        dtype=np.float32,
    )
    self.initial_wrist_roll = float(
        dof_position[dof_names.index("joint6")]
    )
    self.ee_orientation_command = self.initial_ee_orientation.copy()
    # Keep orientation alignment at a fixed, collision-safe point on the
    # robot side of the handle.  Using the measured current EE position
    # as the target lets orientation IK drift move the target itself.
    # Put the open fingertips halfway between the robot-facing panel
    # surface and the handle surface nearest the panel.  This places the
    # tips behind the lever while retaining equal geometric clearance.
    panel_surface = panel_state[:3] + self.panel_half_thickness * self.door_normal
    handle_near_panel = self.grasp_center_pos - handle_radius * self.door_normal
    self.target_tip_mid_world = 0.5 * (panel_surface + handle_near_panel)
    self.target_tip_mid_world[1:] = self.grasp_center_pos[1:]
    self.center_standoff_world = self.target_tip_mid_world + 0.10 * self.door_normal
    if self.shared_robot_geometry is None:
        _introspect_robot_colliders(self, state)
    else:
        self.center_geometry = self.shared_robot_geometry
        _initialize_gripper_limits(self, state)
    # Build the q_open capture frame from the real collision meshes.  All
    # points are expressed in gripper_link coordinates so the desired TCP
    # pose can later be solved by one rigid transform.
    robot_bodies = _as_np(state.get("robot_body_state_w"))
    ee_pos = _as_np(state.get("ee_pos"))[0]
    ee_quat = _as_np(state.get("ee_orn"))[0]
    if robot_bodies is not None and ee_pos is not None and ee_quat is not None:
        fixed_pose = robot_bodies[0, self.fixed_finger_body_idx, :7]
        moving_pose = robot_bodies[0, self.moving_finger_body_idx, :7]
        fixed_points = self.center_geometry["fixed_finger"].world_points(fixed_pose)
        moving_points = self.center_geometry["moving_finger"].world_points(moving_pose)
        fixed_local = _quat_rotate_points_np(_quat_conjugate_np(ee_quat), fixed_points - ee_pos)
        moving_local = _quat_rotate_points_np(_quat_conjugate_np(ee_quat), moving_points - ee_pos)
        # Use the two collider clouds to identify the opposing inner
        # surfaces.  The cloud centroids determine only the sign of the
        # separation axis; all contact points come from the mesh itself.
        separation_local = moving_local.mean(axis=0) - fixed_local.mean(axis=0)
        separation_local /= max(float(np.linalg.norm(separation_local)), 1.0e-8)
        fixed_projection = fixed_local @ separation_local
        moving_projection = moving_local @ separation_local
        fixed_inner_indices = np.flatnonzero(
            fixed_projection >= np.quantile(fixed_projection, 0.85)
        )
        moving_inner_indices = np.flatnonzero(
            moving_projection <= np.quantile(moving_projection, 0.15)
        )
        fixed_inner_center = fixed_local[fixed_inner_indices].mean(axis=0)
        moving_inner_center = moving_local[moving_inner_indices].mean(axis=0)
        capture_center = 0.5 * (fixed_inner_center + moving_inner_center)
        # The insertion/closing axis is the measured robot-facing vector,
        # projected orthogonally to the jaw separation axis.
        depth_local = _quat_rotate_inverse_np(ee_quat, -self.door_normal)
        depth_local -= separation_local * float(np.dot(depth_local, separation_local))
        depth_local /= max(float(np.linalg.norm(depth_local)), 1.0e-8)
        # A fingertip is the distal part of the *inner* collision surface,
        # not the rigid-body origin or the centroid of the whole mesh.
        # Select the last 5 percent along the measured insertion axis.
        fixed_inner_points = fixed_local[fixed_inner_indices]
        moving_inner_points = moving_local[moving_inner_indices]
        fixed_depth = fixed_inner_points @ depth_local
        moving_depth = moving_inner_points @ depth_local
        fixed_tip_points = fixed_inner_points[
            fixed_depth >= np.quantile(fixed_depth, 0.95)
        ]
        moving_tip_points = moving_inner_points[
            moving_depth >= np.quantile(moving_depth, 0.95)
        ]
        fixed_tip_center = fixed_tip_points.mean(axis=0)
        moving_tip_center = moving_tip_points.mean(axis=0)
        tip_capture_center = 0.5 * (fixed_tip_center + moving_tip_center)
        # Seat the lever as deeply as its radius permits.  The proximal
        # edge of either inner jaw limits the circular handle center;
        # retain a small open-jaw clearance so insertion itself does not
        # depend on contact.
        fixed_full_depth = fixed_local @ depth_local
        moving_full_depth = moving_local @ depth_local
        pocket_depth = max(
            float(fixed_full_depth.min()), float(moving_full_depth.min())
        )
        pocket_depth += float(handle_radius) + 0.0015
        pocket_capture_center = capture_center.copy()
        pocket_capture_center += (
            pocket_depth - float(np.dot(pocket_capture_center, depth_local))
        ) * depth_local
        # This point is the handle-center target carried by the TCP.  The
        # grasp frame maps local +X to the door-facing shaft direction,
        # so include the former 10 mm CENTER_HANDLE insertion in the
        # local geometry instead of adding a separate world-frame offset.
        pocket_capture_center[0] -= 0.010
        pocket_depth = float(np.dot(pocket_capture_center, depth_local))
        # A grasp may slide along the lever without becoming invalid.
        # Record the axial band covered by both inner jaw surfaces; the
        # CENTER_HANDLE gate later checks its physical overlap with the
        # finite handle segment instead of forcing both centers to match.
        joint_axis_local = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        fixed_axial = (
            fixed_local[fixed_inner_indices] - pocket_capture_center
        ) @ joint_axis_local
        moving_axial = (
            moving_local[moving_inner_indices] - pocket_capture_center
        ) @ joint_axis_local
        common_axial_min = max(
            float(fixed_axial.min()), float(moving_axial.min())
        )
        common_axial_max = min(
            float(fixed_axial.max()), float(moving_axial.max())
        )
        if common_axial_max <= common_axial_min:
            raise RuntimeError(
                "fixed and moving jaw inner surfaces have no common axial band"
            )
        self.open_template = {
            "fixed_points": fixed_local,
            "moving_points": moving_local,
            "fixed_inner_points": fixed_local[fixed_inner_indices],
            "moving_inner_points": moving_local[moving_inner_indices],
            "depth_axis": depth_local,
            "separation_axis": separation_local,
            "capture_center": capture_center,
            "fixed_tip_center": fixed_tip_center,
            "moving_tip_center": moving_tip_center,
            "tip_capture_center": tip_capture_center,
            "pocket_capture_center": pocket_capture_center,
            "pocket_depth": pocket_depth,
            "common_axial_min": common_axial_min,
            "common_axial_max": common_axial_max,
        }
        all_gripper_points_local = np.concatenate((fixed_local, moving_local), axis=0)
        rotation_sweep_radius = float(
            np.linalg.norm(all_gripper_points_local, axis=1).max()
        )
        self.rotation_sweep_radius = rotation_sweep_radius
        obstacle_front = max(
            float(np.dot(panel_surface, self.door_normal)),
            float(np.dot(self.grasp_center_pos, self.door_normal)) + handle_radius,
        )
        # Keep the aligned arm waypoint far enough from the handle and
        # panel for the complete move from the reset pose to be safe.
        # The refined mount-90 gripper reaches this all-orientations-safe
        # waypoint with a 0.13 m obstacle margin.  The old 0.18 m margin
        # put branch 1 beyond the bounded IK workspace (joint5 saturated),
        # while the full measured rotation_sweep_radius still remains
        # outside the panel and handle envelope.
        safe_ee_projection = obstacle_front + rotation_sweep_radius + 0.13
        self.orientation_standoff_ee_world = ee_pos.copy()
        self.orientation_standoff_ee_world += (
            safe_ee_projection
            - float(np.dot(self.orientation_standoff_ee_world, self.door_normal))
        ) * self.door_normal
        self.orientation_standoff_ee_world[1] = self.grasp_center_pos[1]
        self.orientation_standoff_ee_world[2] = self.grasp_center_pos[2] + 0.05
        initial_offsets_world = _quat_rotate_points_np(
            self.initial_ee_orientation, all_gripper_points_local
        )
        initial_min_normal_offset = float(
            (initial_offsets_world @ self.door_normal).min()
        )
        self.pregrasp_target_world = self.grasp_center_pos.copy()
        pregrasp_projection = obstacle_front + 0.03 - initial_min_normal_offset
        self.pregrasp_target_world += (
            pregrasp_projection
            - float(np.dot(self.pregrasp_target_world, self.door_normal))
        ) * self.door_normal
        self.pregrasp_target_world[2] += 0.05
        self.finger_separation_local = separation_local
        self.finger_closing_local = depth_local
        # The handle must settle in the concave pocket between the two
        # inner jaw surfaces, not at the line joining their distal tips.
        # Recover the TCP pose from that physical pocket center.
        self.grasp_ee_offset_local = pocket_capture_center
        # Keep the measured fingertip geometry for positioning, but do
        # not use its asymmetric centroids to define wrist roll.  The
        # gripper revolute axis is the invariant that must be parallel to
        # the handle axis for the jaws to close in one cross-section.
        tip_sep_local = fixed_tip_center - moving_tip_center
        tip_sep_local /= max(float(np.linalg.norm(tip_sep_local)), 1.0e-8)
        tip_sep_world = _quat_rotate_np(ee_quat, tip_sep_local)
        separation_world = tip_sep_world
        desired_axis_world = -self.door_normal.copy()
        desired_axis_world /= max(
            float(np.linalg.norm(desired_axis_world)), 1.0e-8
        )
        # jointGripper is 86 mm behind the TCP on local -X, so local +X
        # is the physical gripper shaft and local +Y is its revolute
        # axis.  Map those two URDF axes directly instead of inferring a
        # frame from mesh centroids.
        gripper_axis_local = np.array(
            [1.0, 0.0, 0.0], dtype=np.float32
        )
        handle_axis_world = _quat_rotate_np(
            handle_state[3:7], self.handle_axis_local
        )
        handle_axis_world -= desired_axis_world * float(
            np.dot(handle_axis_world, desired_axis_world)
        )
        handle_axis_world /= max(
            float(np.linalg.norm(handle_axis_world)), 1.0e-8
        )

        def _grasp_candidate(joint_axis_world):
            joint_axis_world = np.asarray(joint_axis_world, dtype=np.float32)
            closing_plane_axis = np.cross(
                desired_axis_world, joint_axis_world
            )
            closing_plane_axis /= max(
                float(np.linalg.norm(closing_plane_axis)), 1.0e-8
            )
            return _matrix_to_quat_np(
                np.column_stack(
                    (
                        desired_axis_world,
                        joint_axis_world,
                        closing_plane_axis,
                    )
                )
            )

        # Candidate 1 retains the branch convention used for a free end
        # pointing right; candidate 0 is the 180-degree roll equivalent.
        grasp_candidates = (
            _grasp_candidate(-handle_axis_world),
            _grasp_candidate(handle_axis_world),
        )
        candidate_distances = [
            _quat_distance(ee_quat, candidate) for candidate in grasp_candidates
        ]
        # Mirror the wrist-roll branch with the lever direction.  From the
        # robot-facing view, a lever pointing toward the right hinge must
        # turn the gripper right (branch 1); a lever pointing left must
        # turn it left (branch 0).  Choosing the opposite branch still
        # reaches the handle, but puts the gripper in the wrong mirrored
        # configuration.
        door_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        viewer_right = np.cross(-self.door_normal, door_up)
        viewer_right /= max(float(np.linalg.norm(viewer_right)), 1.0e-8)
        lever_free_end_direction = self.hinge_world - handle_state[:3]
        lever_free_end_direction -= (
            float(np.dot(lever_free_end_direction, self.door_normal)) * self.door_normal
        )
        lever_free_end_direction[2] = 0.0
        lever_free_end_direction /= max(
            float(np.linalg.norm(lever_free_end_direction)), 1.0e-8
        )
        free_end_right = float(np.dot(lever_free_end_direction, viewer_right)) > 0.0
        selected_candidate = 1 if free_end_right else 0
        self.grasp_branch = selected_candidate
        self.grasp_orientation_quat = grasp_candidates[selected_candidate]
        target_gripper_axis_world = _quat_rotate_np(
            self.grasp_orientation_quat, gripper_axis_local
        )
        target_joint_axis_world = _quat_rotate_np(
            self.grasp_orientation_quat,
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        gripper_axis_alignment = float(
            np.dot(target_gripper_axis_world, desired_axis_world)
        )
        joint_axis_alignment = abs(
            float(np.dot(target_joint_axis_world, handle_axis_world))
        )
        if gripper_axis_alignment < 1.0 - 1.0e-5:
            raise RuntimeError(
                "grasp orientation does not align the gripper shaft with "
                f"the door normal: dot={gripper_axis_alignment:.8f}"
            )
        if joint_axis_alignment < 1.0 - 1.0e-5:
            raise RuntimeError(
                "gripper joint axis does not align with the handle axis: "
                f"dot={joint_axis_alignment:.8f}"
            )
        print(
            f"[doorman][geometry] grasp branch={selected_candidate} "
            f"handle_free_end={'right' if free_end_right else 'left'} "
            f"lever_free_end_direction={lever_free_end_direction.round(4).tolist()} "
            f"orientation_distances={np.round(candidate_distances, 4).tolist()} "
            f"gripper_axis_world={target_gripper_axis_world.round(6).tolist()} "
            f"joint_axis_world={target_joint_axis_world.round(6).tolist()} "
            f"axis_alignment={gripper_axis_alignment:.8f} "
            f"joint_axis_alignment={joint_axis_alignment:.8f}",
            flush=True,
        )
        final_offsets_world = _quat_rotate_points_np(
            self.grasp_orientation_quat, all_gripper_points_local
        )
        final_min_normal_offset = float(
            (final_offsets_world @ self.door_normal).min()
        )
        self.final_min_normal_offset = final_min_normal_offset
        final_fixed_offsets_world = _quat_rotate_points_np(
            self.grasp_orientation_quat, fixed_local
        )
        fixed_distal_normal_offset = float(
            (final_fixed_offsets_world @ self.door_normal).min()
        )
        fixed_inner_offset_world = _quat_rotate_np(
            self.grasp_orientation_quat, fixed_inner_center
        )
        fixed_is_upper = bool(fixed_inner_offset_world[2] > 0.0)
        fixed_surface_target_z = float(
            self.grasp_center_pos[2]
            + (handle_radius + 0.013) * (1.0 if fixed_is_upper else -1.0)
        )
        self.fixed_inner_center_local = fixed_inner_center.copy()
        self.fixed_surface_target_z = fixed_surface_target_z
        fixed_anchored_ee_z = float(
            fixed_surface_target_z - fixed_inner_offset_world[2]
        )
        self.grasp_waypoint_world = _effective_grasp_target(
            self,
            self.grasp_center_pos, self.grasp_orientation_quat
        )
        self.grasp_waypoint_world[2] = fixed_anchored_ee_z
        # Both wrist-roll branches use the same collider-derived target.
        # Runtime geometry feedback below compensates IK/load residuals;
        # branch-specific Cartesian constants do not generalize across
        # randomized arm states or handle heights.
        self.center_insert_waypoint_world = self.grasp_waypoint_world.copy()
        self.center_standoff_ee_world = _effective_grasp_target(
            self,
            self.target_tip_mid_world, self.grasp_orientation_quat
        )
        self.center_standoff_ee_world[2] = fixed_anchored_ee_z
        center_standoff_projection = obstacle_front + 0.03 - final_min_normal_offset
        self.center_standoff_ee_world += (
            center_standoff_projection
            - float(np.dot(self.center_standoff_ee_world, self.door_normal))
        ) * self.door_normal
        self.center_standoff_nominal_world = self.center_standoff_ee_world.copy()
        self.center_insert_nominal_world = self.center_insert_waypoint_world.copy()
        self.grasp_nominal_world = self.grasp_waypoint_world.copy()
        # Keep the calculated physical frame for diagnostics, but use a
        # low-level-reachable branch for the first position-only solve.
        # The policy tracks this branch reliably; fingertip geometry is
        # still transformed from the real open-jaw template.
        self.reachable_grasp_orientation_quat = self.grasp_orientation_quat.copy()
        tip_separation_local = tip_sep_local
        insertion_local = depth_local
        self.grasp_ee_offset_world = _quat_rotate_np(self.grasp_orientation_quat, tip_capture_center)
        separation_world = _quat_rotate_np(self.grasp_orientation_quat, tip_separation_local)
        measured_sep_world = separation_world / max(float(np.linalg.norm(separation_world)), 1.0e-8)
        target_sep_world = _quat_rotate_np(self.grasp_orientation_quat, tip_separation_local)
        target_depth_world = _quat_rotate_np(self.grasp_orientation_quat, insertion_local)
        direct_sep_world = moving_points.mean(axis=0) - fixed_points.mean(axis=0)
        direct_sep_world /= max(float(np.linalg.norm(direct_sep_world)), 1.0e-8)
        print(
            f"[doorman][geometry] capture_center_local={capture_center.round(4).tolist()} "
            f"pocket_center_local={pocket_capture_center.round(4).tolist()} "
            f"pocket_depth={pocket_depth:.4f} "
            f"depth_local={depth_local.round(4).tolist()} separation_local={separation_local.round(4).tolist()} "
            f"offset_grasp_world={self.grasp_ee_offset_world.round(4).tolist()} "
            f"measured_sep_world={measured_sep_world.round(4).tolist()} "
            f"direct_sep_world={direct_sep_world.round(4).tolist()} "
            f"target_sep_world={target_sep_world.round(4).tolist()} "
            f"target_depth_world={target_depth_world.round(4).tolist()} "
            f"center_insert_waypoint={self.center_insert_waypoint_world.round(4).tolist()} "
            f"grasp_waypoint={self.grasp_waypoint_world.round(4).tolist()} "
            f"fixed_distal_normal_offset={fixed_distal_normal_offset:.4f}",
            flush=True,
        )
        # Keep an explicit, serializable snapshot of the physical capture
        # geometry.  This is a diagnostic truth source, not an IK target
        # guessed from a rigid-body origin.
        def _bounds(points):
            return {
                "min": points.min(axis=0).astype(float).tolist(),
                "max": points.max(axis=0).astype(float).tolist(),
            }

        fixed_world = fixed_points
        moving_world = moving_points
        fixed_inner_world = _quat_rotate_np(ee_quat, fixed_inner_center) + ee_pos
        moving_inner_world = _quat_rotate_np(ee_quat, moving_inner_center) + ee_pos
        upper_world, lower_world = (
            (fixed_inner_world, moving_inner_world)
            if fixed_inner_world[2] >= moving_inner_world[2]
            else (moving_inner_world, fixed_inner_world)
        )
        handle_center_world = self.grasp_center_pos.copy()
        handle_axis_world = _quat_rotate_np(handle_state[3:7], self.handle_axis_local)
        handle_axis_world /= max(float(np.linalg.norm(handle_axis_world)), 1.0e-8)
        handle_endpoints = [
            (handle_center_world - self.handle_half_length * handle_axis_world).astype(float).tolist(),
            (handle_center_world + self.handle_half_length * handle_axis_world).astype(float).tolist(),
        ]
        self.geometry_report = {
            "handle": {
                "body": self.door_body_names[self.handle_body_idx],
                "collider_path": self.handle_collider_path,
                "center_world": handle_center_world.astype(float).tolist(),
                "axis_world": handle_axis_world.astype(float).tolist(),
                "radius_m": float(handle_radius),
                "half_length_m": float(self.handle_half_length),
                "endpoints_world": handle_endpoints,
                "surface_z_m": [float(self.handle_lower_z), float(self.handle_upper_z)],
            },
            "gripper": {
                "q_open_rad": float(self.q_open),
                "tcp_body": "gripper_link",
                "fixed_finger": {
                    "body": "link06",
                    "collider_paths": self.center_geometry["fixed_finger"].source_paths,
                    "local_bounds": _bounds(self.center_geometry["fixed_finger"].points_body),
                    "world_bounds_at_reset": _bounds(fixed_world),
                    "inner_surface_center_world": fixed_inner_world.astype(float).tolist(),
                },
                "moving_finger": {
                    "body": "gripperMover",
                    "collider_paths": self.center_geometry["moving_finger"].source_paths,
                    "local_bounds": _bounds(self.center_geometry["moving_finger"].points_body),
                    "world_bounds_at_reset": _bounds(moving_world),
                    "inner_surface_center_world": moving_inner_world.astype(float).tolist(),
                },
                "capture_box_at_reset": {
                    "center_world": (0.5 * (upper_world + lower_world)).astype(float).tolist(),
                    "upper_inner_world": upper_world.astype(float).tolist(),
                    "lower_inner_world": lower_world.astype(float).tolist(),
                    "vertical_opening_m": float(abs(upper_world[2] - lower_world[2])),
                    "horizontal_projection_error_m": float(max(
                        np.linalg.norm((upper_world - handle_center_world)[:2]),
                        np.linalg.norm((lower_world - handle_center_world)[:2]),
                    )),
                },
                "tcp_capture_offset_local": pocket_capture_center.astype(float).tolist(),
                "inner_surface_centroid_offset_local": capture_center.astype(float).tolist(),
                "pocket_depth_local_m": float(pocket_depth),
                "tcp_tip_offsets_local": {
                    "fixed_tip": fixed_tip_center.astype(float).tolist(),
                    "moving_tip": moving_tip_center.astype(float).tolist(),
                    "tip_midpoint": tip_capture_center.astype(float).tolist(),
                },
                "finger_depth_ranges_from_tcp_m": {
                    "fixed": [float((fixed_local @ depth_local).min()), float((fixed_local @ depth_local).max())],
                    "moving": [float((moving_local @ depth_local).min()), float((moving_local @ depth_local).max())],
                },
                "joint_gripper_from_tcp": {
                    "position_local_m": [-0.086, 0.0, 0.0],
                    "axis_local": [0.0, 1.0, 0.0],
                    "source": "URDF fixed transforms: link06->stator 0.051 m, stator->joint 0.049 m, stator->TCP 0.135 m",
                },
                "insertion_axis_world": (-self.door_normal).astype(float).tolist(),
                "rotation_sweep_radius_m": rotation_sweep_radius,
                "orientation_standoff_ee_world": self.orientation_standoff_ee_world.astype(float).tolist(),
                "pregrasp_target_ee_world": self.pregrasp_target_world.astype(float).tolist(),
                "center_standoff_ee_world": self.center_standoff_ee_world.astype(float).tolist(),
                "center_insert_ee_world": self.center_insert_waypoint_world.astype(float).tolist(),
            },
        }
        print(
            f"[doorman][geometry] handle endpoints={handle_endpoints} "
            f"r={handle_radius:.4f} half_length={self.handle_half_length:.4f}",
            flush=True,
        )
        print(
            f"[doorman][geometry] q_open={self.q_open:.4f} "
            f"fixed_inner={fixed_inner_world.round(4).tolist()} "
            f"moving_inner={moving_inner_world.round(4).tolist()} "
            f"opening={abs(upper_world[2] - lower_world[2]):.4f}m "
            f"capture_center={(0.5 * (upper_world + lower_world)).round(4).tolist()} "
            f"xy_error={self.geometry_report['gripper']['capture_box_at_reset']['horizontal_projection_error_m']:.4f}m",
            flush=True,
        )
        print(
            f"[doorman][geometry] fingertip offsets from TCP: "
            f"fixed={fixed_tip_center.round(4).tolist()} "
            f"moving={moving_tip_center.round(4).tolist()} "
            f"midpoint={tip_capture_center.round(4).tolist()} "
            f"jointGripper=[-0.086, 0.0, 0.0]",
            flush=True,
        )
    else:
        raise RuntimeError("robot body/EE state unavailable while building the open-gripper collider template")
    _initialize_panel_push_tcp(self)
    # Geometry above is intentionally measured at q_open.  Once that
    # template is cached, start the actual episode with the jaw closed;
    # PREGRASP opens it only after the arm is aligned and stationary.
    gripper_joint_id = int(self.env.gripper_joint_ids[0].item())
    if close_gripper:
        joint_pos = self.env.robot.data.joint_pos.clone()
        joint_vel = self.env.robot.data.joint_vel.clone()
        joint_pos[:, gripper_joint_id] = float(self.q_closed)
        joint_vel[:, gripper_joint_id] = 0.0
        self.env.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self.env.robot.set_joint_position_target(joint_pos)
        self.runner.obs = self.env.get_observations()
        state = self.runner.get_control_state()
    print(
        f"[doorman] RESET physical_handle={self.handle_pos.tolist()} "
        f"grasp_center={self.grasp_center_pos.tolist()} station={self.station_xy.tolist()} "
        f"normal={self.door_normal.tolist()} forward={self.door_forward.tolist()} side={self.robot_side}",
        flush=True,
    )
    return state
