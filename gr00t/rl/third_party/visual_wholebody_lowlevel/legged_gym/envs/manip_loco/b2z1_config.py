from isaaclab.utils import configclass
from legged_gym.envs.base.legged_robot_config import LeggedRobotIsaacLabCfg, ControlCfg
import numpy as np


@configclass
class B2Z1IsaacLabCfg(LeggedRobotIsaacLabCfg):
    task_name = "b2z1"
    action_space = 18
    observation_space = 744
    action_delay = 3
    action_delay_mode = "undelayed"
    action_delay_history_length_min = 4
    action_delay_auto_switch_steps = 240000
    torque_clip = 600.0
    contact_force_threshold = 1.5
    numeric_eps = 1.0e-6
    reset_command = [0.0, 0.0, 0.0]
    obs_scales = {
        "lin_vel": 2.0,
        "ang_vel": 0.25,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
    }
    commands_scale = [2.0, 2.0, 0.25]
    priv_mass_params = [
        1.0417996644973755,
        0.027897033840417862,
        -0.004937552381306887,
        0.0034558435436338186,
        0.004164694342762232,
    ]
    priv_friction_coeffs = [0.520315408706665]
    priv_motor_strength_minus_1 = [
        -0.050519704818725586,
        -0.002183079719543457,
        0.015573859214782715,
        0.08771336078643799,
        0.04324972629547119,
        0.051445960998535156,
        -0.036211252212524414,
        -0.06478011608123779,
        0.02942824363708496,
        0.09396469593048096,
        0.07743573188781738,
        -0.003672182559967041,
    ]
    initial_ee_goal_cart = [0.4619397819042206, 0.0, 0.19134172797203064]

    class paths:
        log_root = "/data/logs"

    decimation = 4
    base_body_name = "base_link"
    gripper_body_name = "gripper_link"
    policy_joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
    ]
    arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    gripper_joint_names = ["jointGripper"]
    foot_body_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    # terminate_body_names = ["base_link", "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh", "FL_calf", "FR_calf", "RL_calf", "RR_calf"]
    terminate_body_names = []
    default_joint_angles = {
        "FL_hip_joint": 0.2, "FL_thigh_joint": 0.8, "FL_calf_joint": -1.5,
        "FR_hip_joint": -0.2, "FR_thigh_joint": 0.8, "FR_calf_joint": -1.5,
        "RL_hip_joint": 0.2, "RL_thigh_joint": 0.8, "RL_calf_joint": -1.5,
        "RR_hip_joint": -0.2, "RR_thigh_joint": 0.8, "RR_calf_joint": -1.5,
        "joint1": 0.0, "joint2": 1.48, "joint3": -0.63, "joint4": -0.84,
        "joint5": 0.0, "joint6": 0.0, "jointGripper": -0.785,
    }
    B2_stiffness = 360
    B2_damping = 5.0
    control = ControlCfg(
        control_type="P",
        # action_scale=0.5,
        action_scale=[
            0.4, 0.45, 0.45,
            0.4, 0.45, 0.45,
            0.4, 0.45, 0.45,
            0.4, 0.45, 0.45,
            2.1, 0.6, 0.6,
            0.0, 0.0, 0.0,
        ],
        clip_actions=100.0,
        clip_observations=100.0,
        stiffness={
            "FL_hip_joint": B2_stiffness, "FL_thigh_joint": B2_stiffness, "FL_calf_joint": B2_stiffness,
            "FR_hip_joint": B2_stiffness, "FR_thigh_joint": B2_stiffness, "FR_calf_joint": B2_stiffness,
            "RL_hip_joint": B2_stiffness, "RL_thigh_joint": B2_stiffness, "RL_calf_joint": B2_stiffness,
            "RR_hip_joint": B2_stiffness, "RR_thigh_joint": B2_stiffness, "RR_calf_joint": B2_stiffness,
            "joint1": 5.0, "joint2": 5.0, "joint3": 5.0, "joint4": 5.0, "joint5": 5.0, "joint6": 5.0,
            "jointGripper": 5.0,
        },
        damping={
            "FL_hip_joint": B2_damping, "FL_thigh_joint": B2_damping, "FL_calf_joint": B2_damping,
            "FR_hip_joint": B2_damping, "FR_thigh_joint": B2_damping, "FR_calf_joint": B2_damping,
            "RL_hip_joint": B2_damping, "RL_thigh_joint": B2_damping, "RL_calf_joint": B2_damping,
            "RR_hip_joint": B2_damping, "RR_thigh_joint": B2_damping, "RR_calf_joint": B2_damping,
            "joint1": 0.5, "joint2": 0.5, "joint3": 0.5, "joint4": 0.5, "joint5": 0.5, "joint6": 0.5,
            "jointGripper": 0.5,
        },
    )

    class env:
        num_envs = 6144
        num_actions = 12 + 6 #CAUTION
        num_torques = 12 + 6
        action_delay = 3  # -1 for no delay
        action_delay_mode = "auto"  # auto: keep training curriculum, undelayed: latest action, delayed: one-step delayed action
        ee_goal_obs_mode = "command"  # command: use sampled EE command directly, arm_base_target: use target relative to arm base
        num_gripper_joints = 1
        num_proprio = 2 + 3 + 18 + 18 + 12 + 4 + 3 + 3 + 3
        num_priv = 5 + 1 + 12
        history_len = 10
        num_observations = num_proprio * (history_len + 1) + num_priv
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        send_timeouts = True # send time out information to the algorithm
        reorder_dofs = True
        teleop_mode = False # Overriden in teleop.py. When true, commands come from keyboard
        teleop_arm_control_mode = "ee"
        teleop_debug = True
        teleop_hold_actual_ee_target_on_init = True
        teleop_input_regularization = False # If true, preprocess teleop inputs before feeding the policy/control stack
        teleop_keyboard_control = True
        teleop_auto_height_reference = True
        teleop_zero_lin_vel_x_clip = 0.2
        teleop_zero_lin_vel_y_clip = 0.2
        teleop_zero_ang_vel_yaw_clip = 0.5
        teleop_lin_vel_x_limit = 0.8
        teleop_lin_vel_y_limit = 0.8
        teleop_ang_vel_yaw_limit = 1.0
        teleop_ee_goal_x_limit = [-0.5, 1.0]
        teleop_ee_goal_y_limit = [-0.7, 0.7]
        teleop_ee_goal_z_limit = [-0.6, 0.6]
        teleop_restore_arm_gripper_state_on_reset = False
        teleop_key_repeat_delay_s = 0.35
        teleop_key_repeat_rate_hz = 6.0
        teleop_base_lin_vel_step = 0.05
        teleop_base_ang_vel_step = 0.05
        teleop_arm_joint_step = 0.05
        teleop_ee_goal_pos_step = 0.05
        teleop_ee_goal_orn_step = 0.05
        teleop_gripper_step = 0.05
        record_video = False
        stand_by = False
        stop_update_goal = False
        observe_gait_commands = False
        observe_foot_contacts = True
        zero_observed_foot_contacts = True
        gait_pattern = "adaptive_trot"
        trot_stance_ratio = 0.5
        trot_swing_ratio = 0.5
        trot_swing_duration_max_s = 0.4
        fixed_trot_frequency = 1.5
        gait_max_stride_x = 0.4
        gait_max_stride_y = 0.2
        gait_transition_duration_s = 0.5
        trunk_follow_arm_obs_mode = "default"

    class goal_ee:
        collision_upper_limits = [0.2, 0.2, -0.05]
        collision_lower_limits = [-0.7, -0.2, -0.7]
        underground_limit = -0.7
        num_collision_check_samples = 10
        arm_induced_pitch = 0.38
        traj_time = [1, 3]
        hold_time = [0.5, 2]
        yaw_adaptive_time_ref = 1.2
        yaw_adaptive_time_min_scale = 1.0
        yaw_adaptive_time_max_scale = 5.0
        ik_damping = 0.05
        traj_timesteps_default = 1.0
        traj_total_timesteps_default = 1.0e9
        command_mode = "sphere"
        sphere_error_scale = [1, 1, 1]
        orn_error_scale = [1, 1, 1]

        class ranges:
            init_pos_start = [0.5, np.pi / 8, 0]
            init_pos_end = [0.7, 0, 0]
            ee_goal_sampling_mode = "arm_front_sphere"
            pos_l = [0.4, 0.95]
            pos_p = [-1 * np.pi / 2.1, 1 * np.pi / 3]
            pos_y = [-1.2, 1.2]
            omnidirectional_init_pos_y = [-5 * np.pi / 6, 5 * np.pi / 6]
            omnidirectional_rear_transition_pos_y_abs = 2 * np.pi / 3
            omnidirectional_pos_l = [0.2, 0.95]
            omnidirectional_rear_pos_l = [0.2, 0.5]
            omnidirectional_rear_pos_p = [-np.pi / 6, np.pi / 6]
            delta_orn_r = [-0.5, 0.5]
            delta_orn_p = [-0.5, 0.5]
            delta_orn_y = [-0.5, 0.5]

        class body_importance_sampling:
            extension = 0.2
            back_start_offset = 0.1
            side_height = [0.0, 1.0]
            ground_height = [0.0, 0.1]
            ground_height_ratio = 0.25
            back_height = [0.0, 0.3]

        class urdf_mount:
            arm_base_offset = [0.2, 0.0, 0.09]
            mount_yaw_offset = 0.0
            arm_waist_offset_z = 0.0585
            arm_shoulder_offset_z = 0.045

        class sphere_center:
            x_offset = 0.2
            y_offset = 0.0
            z_invariant_offset = 0.7
            mixed_height_reference = False # If false, omit the policy mode bit, but keep both height-reference modes available for deployment.
            trunk_follow_ratio = 0.5 # Fraction of trunk-following goal episodes when mixed_height_reference is enabled; otherwise treated as 0 during training.
            trunk_follow_anchor = "arm_waist"

    class xr_teleop:
        ros_topic = "/xr_pose"
        active_status = 3
        axis_deadband = 0.08
        max_lin_vel_x = 0.8
        max_lin_vel_y = 0.8
        max_yaw_rate = 1.0
        pose_tracking_button = "gripper"
        pose_tracking_threshold = 0.4
        stale_timeout_s = 0.35
        bimanual_mode = False
        head_position_filter_alpha = 0.2
        min_hand_radius = 0.08
        min_arm_radius = 0.08
        max_arm_radius = 0.95
        min_arm_z = -0.7
        max_arm_z = 0.85
        arm_joint_motion_timeout_s = 7.0
        front_preset_q = ""
        back_preset_q = ""
        right_controller_pose_offset_xyz = "0,0.03,-0.015"
        right_controller_pose_offset_rpy_deg = "-35,-6,8"

    class rewards:
        reward_scale_preset = "legacy"
        reward_container_name = ""
        only_positive_rewards = False
        soft_dof_pos_limit = 1.0
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 0.4
        kappa_gait_probs = 0.07
        min_contact_force = 10.0
        feet_height_target = 0.2
        swing_ratio = 0.375
        stance_ratio = 0.625
        clearance_height_target = -0.3
        feet_airtime_allfeet = False
        feet_height_allfeet = False

        # legacy 和 height_flexible 模式的参数
        base_height_target = 0.55
        base_height_target_min = 0.40
        base_height_target_max = 0.70
        max_contact_force = 200.0
        gait_transition_lower = 0.1
        gait_transition_upper = 0.9
        crouch_hip_delta = 0.0
        crouch_thigh_delta = 0.35
        crouch_calf_delta = -0.55
        tiptoe_hip_delta = 0.0
        tiptoe_thigh_delta = -0.5565
        tiptoe_calf_delta = 0.8995

        # robot_lab Unitree B2 模式的参数
        robotlab_undesired_contacts_threshold = 1.0
        robotlab_contact_force_threshold = 100.0
        robotlab_stand_still_scale = 3.0
        robotlab_velocity_threshold = 0.5
        robotlab_command_threshold = 0.1
        robotlab_feet_height_body_target = -0.4
        robotlab_feet_height_tanh_mult = 2.0
        robotlab_feet_height_follows_ground = True
        robotlab_feet_height_world_clearance = (
            base_height_target + robotlab_feet_height_body_target
        )

        ee_goal_arm_base_height_threshold = 0.4
        feet_airtime_allfeet = True
        feet_height_allfeet = True

        # Reward sigmas
        tracking_lin_vel_xy_exp_l2_sigma = 0.5
        tracking_lin_vel_x_exp_l1_sigma = 0.5
        tracking_lin_vel_y_exp_l2_sigma = 0.5
        tracking_lin_vel_zero_cmd_exp_l1_sigma = 1.0
        tracking_ang_vel_yaw_exp_l1_sigma = 0.5
        tracking_ang_vel_yaw_exp_l2_sigma = 0.5
        tracking_ee_exp_l1_sigma = 1.0
        tracking_ee_world_exp_l1_sigma = 0.5
        foot_vel_exp_l2_sigma = 1.0
        foot_force_exp_l2_sigma = 10.0
        leg_posture_exp_l1_sigma = 20.0
        dof_default_pos_exp_l1_sigma = 20.0
        robotlab_tracking_exp_l2_sigma = 0.5
        base_height_exp_l2_sigma = 0.1 ** 0.5
        dof_posture_exp_l2_sigma = 2.0 ** 0.5
        sync_sigma = 1.0  # currently not used by any reward implementation

        class scales:
            tracking_contacts_shaped_force_exp_l2 = 0.0 # 惩罚摆动足触地力过大
            tracking_contacts_shaped_vel_exp_l2 = 0.0 # 惩罚支撑足滑动过快
            tracking_contacts_shaped_force_weighted_exp_l2 = 1.0 # 惩罚摆动足触地力过大（按期望接触状态加权）
            tracking_contacts_shaped_vel_weighted_exp_l2 = 1.0 # 惩罚支撑足滑动过快（按期望接触状态加权）
            feet_air_time = 0.0 # 奖励迈步腾空更久
            feet_air_time_contact_filt_04 = 0.8 # 奖励迈步腾空更久（contact_filt、0.4s 阈值版本）
            feet_height = 0.0 # 惩罚摆腿抬脚不足
            feet_height_mse = 1.0 # 惩罚摆腿高度偏离目标高度（平方误差版本）
            feet_height_mse_gait = 0.0 # 按步态相位加权的摆动足高度平方误差
            feet_height_standing_clearance = 1.0 # 惩罚静止时足端离地过高
            feet_height_turning_mse = 3.0 # 惩罚转向时前脚高度偏离目标高度
            tracking_lin_vel_x_ratio_zero_cmd_exp_l1 = 1.0 # 奖励前向速度跟踪
            tracking_lin_vel_y_ratio_zero_cmd_exp_l1 = 0.0 # 奖励侧向速度跟踪
            tracking_lin_vel_x_l1 = 0.0 # 奖励前向速度贴近命令
            tracking_lin_vel_x_exp_l1 = 0.0 # 奖励前向速度指数跟踪
            tracking_ang_vel_yaw_exp_l2 = 0.5 # 奖励偏航角速度跟踪
            ee_goal_arm_base_height_violation = 0.0 # 惩罚EE目标高度偏离机身机械臂基座高度过大
            penalty_lin_vel_y = 0.0 # 惩罚侧向漂移速度
            stand_still_dof_exp_l1 = 1.0 # 奖励静止时站稳
            stand_still_flexible_dof_exp_l1 = 0.0 # 奖励静止时站稳（Height-flexible）
            walking_dof_exp_l1 = 1.5 # 奖励行走时关节规整
            walking_dof_flexible_exp_l1 = 0.0 # 奖励行走时关节规整（Height-flexible）
            alive = 0.0 # 奖励回合持续存活
            lin_vel_z = -3.0 # 惩罚机身上下晃动
            lin_vel_y_square_penalty = 0.0 # 惩罚侧向速度平方
            roll = -1.5 # 惩罚机身横滚倾斜
            pitch = 0.0 # 惩罚机身俯仰倾斜
            orientation = -0.2 # 惩罚机身姿态倾斜
            ang_vel_xy = -0.02 # 惩罚机身横俯角速度
            orientation_walking = 0.0 # 惩罚行走时姿态倾斜
            orientation_standing = 0.0 # 惩罚站立时姿态倾斜
            base_height = 0.0 # 惩罚机身高度偏差
            base_height_exp_l2 = 3.0 # 奖励机身高度指数跟踪
            base_height_nominal = 0.0 # 在允许区间内弱偏好回到默认高度（Height-flexible）
            base_height_band = 0.0 # 惩罚机身高度偏差（Height-flexible）
            base_height_walking = 0.0 # 惩罚行走时机身高度偏差
            base_height_standing = 0.0 # 惩罚站立时机身高度偏差
            base_height_band_exp_l2 = 0.0 # 奖励机身高度保持在允许区间内（指数版本）
            shared_height_posture_exp_l2 = 0.0 # 奖励四腿处在共享高度姿态流形上（指数版本）
            hip_pos = -0.5 # 惩罚髋关节偏离默认
            hip_pos_standing = -0.1 # 惩罚静止时髋关节偏离默认
            hip_pos_flexible = 0.0 # 惩罚髋关节偏离默认（Height-flexible）
            thigh_pos = -0.1 # 惩罚大腿关节偏离默认
            thigh_pos_back = -1.5 # 惩罚后腿大腿关节偏离默认范围
            calf_pos = -1.0 # 惩罚小腿关节偏离默认范围
            dof_default_pos_exp_l1 = 0.0 # 奖励关节贴近默认位
            dof_error = 0.0 # 惩罚关节默认位误差
            dof_default_pos_exp_l2 = 0.05 # 奖励关节贴近默认位（指数版本）
            dof_vel_abs = -0.0008 # 惩罚关节速度绝对值过大
            action_rate = -0.05 # 惩罚动作变化过快
            dof_acc = -5e-7 # 惩罚关节加速度过大
            dof_pos_limits = -3.0 # 惩罚关节逼近限位
            delta_torques = -1.0e-7 / 4.0 # 惩罚扭矩突变过大
            torques = -2.5e-7 # 惩罚关节扭矩过大
            torques_walking = 0.0 # 惩罚行走时扭矩过大
            torques_standing = 0.0 # 惩罚站立时扭矩过大
            work = 0.0 # 惩罚腿部净做功大
            energy_square = 0.0 # 惩罚腿部平方能耗
            energy_square_walking = 0.0 # 惩罚行走时平方能耗
            energy_square_standing = 0.0 # 惩罚站立时平方能耗
            collision = -5.0 # 惩罚机身发生碰撞
            feet_jerk = 0.0 # 惩罚足端冲击突变
            feet_drag = -0.08 # 惩罚足端拖地滑动
            feet_contact_forces = -0.01 # 惩罚足端接触力过大
            feet_contact_forces_standing = -0.001 # 惩罚静止时足端接触力不足

            robotlab_lin_vel_z_l2 = 0.0 # robot_lab：惩罚机身 z 方向线速度平方
            robotlab_ang_vel_xy_l2 = 0.0 # robot_lab：惩罚机身横滚/俯仰角速度平方
            robotlab_joint_torques_l2 = 0.0 # robot_lab：惩罚关节扭矩平方
            robotlab_joint_acc_l2 = 0.0 # robot_lab：惩罚关节加速度平方
            robotlab_joint_pos_limits = 0.0 # robot_lab：惩罚关节接近或超过位置限位
            robotlab_joint_power = 0.0 # robot_lab：惩罚关节功率绝对值 |tau * qd|
            robotlab_default_posture_standing = 0.0 # robot_lab：静止时惩罚四腿偏离固定默认站姿
            robotlab_default_posture_all_phase = 0.0 # robot_lab：全程惩罚四腿偏离固定默认站姿，静止时加权增强
            robotlab_shared_height_posture_standing = 0.0 # robot_lab：静止时惩罚四腿偏离共享高度联合姿态流形
            robotlab_shared_height_posture_all_phase = 0.0 # robot_lab：全程惩罚四腿偏离共享高度联合姿态流形，静止时加权增强
            robotlab_joint_mirror = 0.0 # robot_lab：惩罚对角腿关节姿态不镜像
            robotlab_action_rate_l2 = 0.0 # robot_lab：惩罚动作变化平方
            robotlab_undesired_contacts = 0.0 # robot_lab：惩罚非足端刚体发生接触
            robotlab_contact_forces = 0.0 # robot_lab：惩罚足端接触力超过阈值
            robotlab_track_lin_vel_xy_exp_l2 = 0.0 # robot_lab：奖励 xy 平面线速度指数跟踪
            robotlab_track_ang_vel_z_exp_l2 = 0.0 # robot_lab：奖励偏航角速度指数跟踪
            robotlab_feet_contact_without_cmd = 0.0 # robot_lab：无运动命令时奖励足端接触
            robotlab_feet_height_body = 0.0 # robot_lab：惩罚足端在机体系下偏离目标高度
            robotlab_upward = 0.0 # robot_lab：奖励机身朝上
            robotlab_clock_swing_force_exp_l2 = 0.0 # robot_lab-clock：跟随clock，惩罚摆动足触地力过大
            robotlab_clock_stance_vel_exp_l2 = 0.0 # robot_lab-clock：跟随clock，惩罚支撑足滑动过快
            robotlab_clock_stance_contact_exp_l2 = 0.0 # robot_lab-clock：跟随clock，轻微惩罚支撑足接触力不足

        class scale_presets:
            legacy = {
                "stand_still_dof_exp_l1": 1.0,
                "stand_still_flexible_dof_exp_l1": 0.0,
                "walking_dof_exp_l1": 1.5,
                "walking_dof_flexible_exp_l1": 0.0,
                "roll": -1.5,
                "pitch": 0.0,
                "ang_vel_xy": -0.02,
                "orientation": -0.2,
                "base_height_exp_l2": 3.0,
                "base_height_band_exp_l2": 0.0,

                "thigh_pos": -0.1,
                "thigh_pos_back": -1.5,
                "calf_pos": -1.0,
                "dof_default_pos_exp_l2": 0.05,

                "lin_vel_z": -3.0,

                "tracking_contacts_shaped_force_weighted_exp_l2": 1.0,
                "tracking_contacts_shaped_vel_weighted_exp_l2": 1.0,
                "tracking_contacts_shaped_force_exp_l2": 0.0,
                "tracking_contacts_shaped_vel_exp_l2": 0.0,
            }
            height_flexible = {
                "stand_still_dof_exp_l1": 0.0,
                "stand_still_flexible_dof_exp_l1": 0.0,
                "walking_dof_exp_l1": 0.0,
                "walking_dof_flexible_exp_l1": 0.0,
                "roll": -1.5,
                "pitch": -2.0,
                "ang_vel_xy": -0.4,
                "orientation": -6.0,
                "base_height_exp_l2": 0.0,
                "base_height_nominal": -1.0,
                "base_height_band_exp_l2": 3.0,

                "thigh_pos": 0.0,
                "thigh_pos_back": 0.0,
                "calf_pos": 0.0,
                "dof_default_pos_exp_l2": 0.0,

                "lin_vel_z": -0.2,

                "tracking_contacts_shaped_force_weighted_exp_l2": 0.0,
                "tracking_contacts_shaped_vel_weighted_exp_l2": 0.0,
                "tracking_contacts_shaped_force_exp_l2": -1.0,
                "tracking_contacts_shaped_vel_exp_l2": -1.0,

                "feet_height_mse": 0.0,
                "feet_height_turning_mse": 0.0,
                "feet_height_mse_gait": 3.0,
                "ee_goal_arm_base_height_violation": -1.0,
                "shared_height_posture_exp_l2": 1.0,
            }
            robotlab_b2 = {
                "__zero_all_non_robotlab_rewards__": True,
                "base_height_nominal": -4.0,
                "orientation": -10.0,
                "robotlab_lin_vel_z_l2": -2.0,
                "robotlab_ang_vel_xy_l2": -0.05,
                "robotlab_upward": 3.0,
                "robotlab_track_lin_vel_xy_exp_l2": 3.0,
                "robotlab_track_ang_vel_z_exp_l2": 1.5,
                "robotlab_joint_torques_l2": -1e-5,
                "robotlab_joint_acc_l2": -3e-7,
                "robotlab_joint_pos_limits": -5.0,
                "robotlab_joint_power": -1e-5,
                "robotlab_default_posture_standing": 0.0,
                "robotlab_default_posture_all_phase": -0.5,
                "robotlab_shared_height_posture_standing": 0.0,
                "robotlab_shared_height_posture_all_phase": -4.0,
                "robotlab_joint_mirror": -0.05,
                "robotlab_action_rate_l2": -0.01,
                "robotlab_undesired_contacts": -1.0,
                "robotlab_contact_forces": -1.5e-4,
                "robotlab_feet_contact_without_cmd": 0.5,
                "robotlab_feet_height_body": -5.0,
                "robotlab_clock_swing_force_exp_l2": -4.0,
                "robotlab_clock_stance_vel_exp_l2": -4.0,
                "robotlab_clock_stance_contact_exp_l2": -0.5,
            }

        _reward_scale_presets = {
            "legacy": scale_presets.legacy,
            "height_flexible": scale_presets.height_flexible,
            "robotlab_b2": scale_presets.robotlab_b2,
        }
        if reward_scale_preset not in _reward_scale_presets:
            raise ValueError(f"Unsupported rewards.reward_scale_preset={reward_scale_preset}")
        _selected_reward_scale_preset = dict(_reward_scale_presets[reward_scale_preset])
        if _selected_reward_scale_preset.pop("__zero_all_non_robotlab_rewards__", False):
            for _reward_name in dir(scales):
                if _reward_name.startswith("_") or _reward_name.startswith("robotlab_"):
                    continue
                _reward_scale = getattr(scales, _reward_name)
                if isinstance(_reward_scale, (int, float)):
                    setattr(scales, _reward_name, 0.0)
        for _reward_name, _reward_scale in _selected_reward_scale_preset.items():
            setattr(scales, _reward_name, _reward_scale)
        del _reward_scale_presets, _selected_reward_scale_preset, _reward_name, _reward_scale

        class only_positive_rewards_presets:
            legacy = True
            height_flexible = True
            robotlab_b2 = False

        only_positive_rewards = only_positive_rewards_presets.__dict__[reward_scale_preset]

        class arm_scales:
            arm_termination = None # 惩罚机械臂回合终止
            tracking_ee_sphere_exp_l1 = 0.0 # 奖励末端球坐标跟踪
            tracking_ee_world_exp_l1 = 0.8 # 奖励末端世界坐标跟踪
            tracking_ee_sphere_walking_exp_l1 = 0.0 # 奖励行走时末端球坐标跟踪
            tracking_ee_sphere_standing_exp_l1 = 0.0 # 奖励站立时末端球坐标跟踪
            tracking_ee_cart_exp_l1 = None # 奖励末端笛卡尔跟踪
            arm_energy_abs_sum = None # 惩罚机械臂能耗过大
            tracking_ee_orn_exp_l1 = 0.0 # 奖励末端姿态跟踪
            tracking_ee_orn_ry_exp_l1 = None # 奖励末端滚偏姿态跟踪

        class arm_scale_presets:
            legacy = {
                "tracking_ee_world_exp_l1": 3.0,
            }
            height_flexible = {
                "tracking_ee_world_exp_l1": 3.0,
            }
            robotlab_b2 = {
                "tracking_ee_world_exp_l1": 5.0,
            }

        _arm_reward_scale_presets = {
            "legacy": arm_scale_presets.legacy,
            "height_flexible": arm_scale_presets.height_flexible,
            "robotlab_b2": arm_scale_presets.robotlab_b2,
        }
        _selected_arm_reward_scale_preset = _arm_reward_scale_presets[reward_scale_preset]
        for _arm_reward_name, _arm_reward_scale in _selected_arm_reward_scale_preset.items():
            setattr(arm_scales, _arm_reward_name, _arm_reward_scale)
        del _arm_reward_scale_presets, _selected_arm_reward_scale_preset, _arm_reward_name, _arm_reward_scale


    class train:
        max_iterations = 10000
        save_interval = 100
        train_log_every = 100
        num_steps_per_env = 24
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        leg_control_head_hidden_dims = [128, 128]
        arm_control_head_hidden_dims = [128, 128]
        priv_encoder_dims = [64, 20]
        activation = "elu"
        init_std = 1.0
        num_leg_actions = 12
        num_arm_actions = 6
        adaptive_arm_gains = False
        adaptive_arm_gains_scale = 1.0
        output_tanh = False
        num_learning_epochs = 5
        num_mini_batches = 4
        clip_param = 0.2
        gamma = 0.998
        lam = 0.95
        value_loss_coef = 1.0
        entropy_coef = 0.0
        learning_rate = 1.0e-3
        max_grad_norm = 1.0
        use_clipped_value_loss = True
        schedule = "adaptive"
        desired_kl = 0.01
        mixing_schedule = [0.5, 2000, 4000]
        torque_supervision = False
        torque_supervision_schedule = [0.1, 1000, 1000]
        min_policy_std = [0.05] * 18
        dagger_update_freq = 20
        priv_reg_coef_schedule = [0.0, 0.0, 0.0]

    class commands:
        curriculum = True
        num_commands = 3
        resampling_time = 3.0 # time before command are changed[s]

        # Command-range curricula
        lin_vel_x_min_schedule = [0.0, -0.8, 5000, 5000]
        lin_vel_x_max_schedule = [0.8, 0.8, 0, 0]
        lin_vel_y_schedule = [0.0, 0.0, 0, 0]
        ang_vel_yaw_schedule = [1.0, 1.0, 0, 0]
        non_omni_pos_y_schedule = [1.2, 1.2, 0, 0]
        curriculum_playback_counter = 0.0
        curriculum_playback_total_iterations = None
        ang_vel_yaw_clip = 0.5
        lin_vel_x_clip = 0.2
        lin_vel_y_clip = 0.2

    class asset():
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/b2z1/urdf/b2z1.urdf"
        base_name = "base_link"
        gripper_name = "gripper_link"
        arm_waist_name = "joint1"
        hip_joint_names = ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"]
        policy_leg_joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ]
        policy_foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        penalize_contacts_on = ["thigh", "base_link", "calf"]
        mount_urdf_generator = "b2z1"
        robot_ablation = "none"
        leg_collision_scale = 1.0
        
    def __post_init__(self):
        super().__post_init__()

        # IsaacLab reads robot.init_state.joint_pos for initial joint positions.
        # default_joint_angles also feeds the control logic.
        self.robot.init_state.pos = (0.0, 0.0, 0.5)
        self.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)  # IsaacLab uses wxyz
        # self.robot.init_state.rot = (0.70710678, -0.70710678, 0.0, 0.0)  # IsaacLab uses wxyz 
        self.robot.init_state.joint_pos = dict(self.default_joint_angles)    


def apply_reward_scale_preset(cfg: B2Z1IsaacLabCfg, preset: str):
    preset = str(preset or "legacy")
    presets = {
        "legacy": cfg.rewards.scale_presets.legacy,
        "height_flexible": cfg.rewards.scale_presets.height_flexible,
        "robotlab_b2": cfg.rewards.scale_presets.robotlab_b2,
    }
    if preset not in presets:
        raise ValueError(
            f"Unsupported reward_scale_preset={preset!r}. "
            f"Expected one of: {', '.join(sorted(presets))}"
        )
    cfg.rewards.reward_scale_preset = preset
    selected_preset = dict(presets[preset])
    if selected_preset.pop("__zero_all_non_robotlab_rewards__", False):
        for name in dir(cfg.rewards.scales):
            if name.startswith("_") or name.startswith("robotlab_"):
                continue
            value = getattr(cfg.rewards.scales, name)
            if isinstance(value, (int, float)):
                setattr(cfg.rewards.scales, name, 0.0)
    for name, value in selected_preset.items():
        setattr(cfg.rewards.scales, name, value)
    if hasattr(cfg.rewards, "arm_scale_presets"):
        arm_presets = {
            "legacy": cfg.rewards.arm_scale_presets.legacy,
            "height_flexible": cfg.rewards.arm_scale_presets.height_flexible,
            "robotlab_b2": cfg.rewards.arm_scale_presets.robotlab_b2,
        }
        for name, value in arm_presets[preset].items():
            setattr(cfg.rewards.arm_scales, name, value)
    if hasattr(cfg.rewards, "only_positive_rewards_presets"):
        cfg.rewards.only_positive_rewards = bool(
            cfg.rewards.only_positive_rewards_presets.__dict__[preset]
        )
