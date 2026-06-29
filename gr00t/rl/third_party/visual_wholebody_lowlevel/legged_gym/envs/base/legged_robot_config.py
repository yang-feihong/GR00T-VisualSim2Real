from __future__ import annotations
import os
from dataclasses import MISSING, field
from pathlib import Path

from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg


_LOW_LEVEL_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_B2Z1_URDF = _LOW_LEVEL_ROOT / "resources/robots/b2z1/urdf/b2z1_isaacsim_mesh_axis_fixed.urdf"


@configclass
class ControlCfg:
    control_type: str = "P"
    action_scale: float = 0.5
    stiffness: dict = MISSING
    damping: dict = MISSING
    clip_actions: float = 100.0
    clip_observations: float = 100.0


@configclass
class CommandCfg:
    num_commands: int = 3
    resampling_time: float = 3.0
    lin_vel_x: tuple = (-0.8, 0.8)
    lin_vel_y: tuple = (0.0, 0.0)
    ang_vel_yaw: tuple = (-1.0, 1.0)


@configclass
class ViewerControlCfg:
    ref_env: int = 0
    pos: list = field(default_factory=lambda: [10.0, 0.0, 6.0])
    lookat: list = field(default_factory=lambda: [11.0, 5.0, 3.0])
    follow_vec_local: list = field(default_factory=lambda: [0.0, 2.0, 1.0])
    follow_yaw_update_threshold_deg: float = 20.0
    orbit_radius_min: float = 0.5
    orbit_pitch_limit_deg: float = 85.0
    orbit_yaw_step_deg: float = 5.0
    orbit_pitch_step_deg: float = 5.0
    orbit_radius_step: float = 0.2


@configclass
class VideoCfg:
    fps: int = 25
    render_envs: int = 5
    output_root: str = "../../logs/videos"


LegacyControlCfg = ControlCfg
LegacyCommandCfg = CommandCfg
LegacyViewerCfg = ViewerControlCfg
LegacyVideoCfg = VideoCfg


@configclass
class LeggedRobotIsaacLabCfg(DirectRLEnvCfg):
    """DirectRLEnvCfg for the legged robot IsaacLab environment."""
    # RL timing
    decimation: int = 4
    episode_length_s: float = 10.0
    action_space: int = 12
    observation_space: int = 235
    state_space: int = 0

    # simulation / scene
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=4,
            max_velocity_iteration_count=0,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=True)

    # Terrain importer configuration.
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        env_spacing=3.0,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
    )

    # Asset path. Override with --robot_urdf_path or edit here.
    robot_urdf_path: str = os.environ.get(
        "LEGGED_GYM_ROBOT_URDF",
        str(_DEFAULT_B2Z1_URDF),
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path="",  # patched at runtime from cfg.robot_urdf_path
            activate_contact_sensors=True,
            force_usd_conversion=True,
            fix_base=False,
            merge_fixed_joints=True,
            replace_cylinders_with_capsules=True,
            self_collision=False,
            make_instanceable=False,            
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                target_type="none",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0,
                    damping=0.0,
                ),    
            ),        
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.55), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}),
        actuators={
            # Legs use manual torque PD.
            # Keep stiffness/damping 0 to avoid double PD.
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*hip_joint", ".*thigh_joint", ".*calf_joint"],
                effort_limit_sim=600.0,
                velocity_limit_sim=100.0,
                stiffness=0,
                damping=0,
            ),            
            # "legs": ImplicitActuatorCfg(
            #     joint_names_expr=[".*hip_joint", ".*thigh_joint", ".*calf_joint"],
            #     effort_limit_sim=600.0,
            #     velocity_limit_sim=100.0,
            #     stiffness={
            #         ".*hip_joint": 500.0,
            #         ".*thigh_joint": 500.0,
            #         ".*calf_joint": 500.0,
            #     },
            #     damping={
            #         ".*hip_joint": 20.0,
            #         ".*thigh_joint": 20.0,
            #         ".*calf_joint": 20.0,
            #     },
            # ),

            # Z1 arm + gripper uses position target drive.
            # Give IsaacLab drive stiffness/damping here.
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[
                    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "jointGripper",
                ],
                effort_limit_sim=80.0,
                velocity_limit_sim=20.0,
                stiffness={
                    "joint1": 400.0,
                    "joint2": 400.0,
                    "joint3": 400.0,
                    "joint4": 400.0,
                    "joint5": 400.0,
                    "joint6": 400.0,
                    "jointGripper": 400.0,
                },
                damping={
                    "joint1": 40.0,
                    "joint2": 40.0,
                    "joint3": 40.0,
                    "joint4": 40.0,
                    "joint5": 40.0,
                    "joint6": 40.0,
                    "jointGripper": 40.0,
                },
            ),
        }
    )
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    control: ControlCfg = ControlCfg(stiffness={}, damping={})
    commands: CommandCfg = CommandCfg()
    viewer_control: ViewerControlCfg = ViewerControlCfg()
    video: VideoCfg = VideoCfg()
    default_joint_angles: dict = MISSING
    policy_joint_names: list = MISSING
    foot_body_names: list = MISSING
    terminate_body_names: list = MISSING
    base_body_name: str = "base_link"
    gripper_body_name: str = "gripper_link"
    num_gripper_joints: int = 1
    send_timeouts: bool = True
    enable_height_scan: bool = False
    enable_contact_sensor: bool = True
    compute_rewards: bool = True
    profile_env_step: bool = False
    check_terminations: bool = True
    scene_usd_path: str = ""
    scene_prim_path: str = "/World/ImportedScene"
    scene_position: list = field(default_factory=list)
    rgb_camera_specs: list = field(default_factory=list)
    rgb_camera_draw_in_viewer: bool = False
    rgb_camera_backend: str = "auto"
    rgb_camera_update_period_s: float = 0.1
