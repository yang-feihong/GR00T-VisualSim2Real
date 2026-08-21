from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import torch

LOW_LEVEL_ROOT = Path(__file__).resolve().parents[2]
for p in [str(LOW_LEVEL_ROOT)]:
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

from legged_gym.utils import isaaclab_app, run_metadata
from legged_gym.utils.b2z1_mount import MOUNT_URDF_SPECS, ensure_mount_urdf

OBSERVATION_BASE_PROPRIO = 2 + 3 + 18 + 18 + 12 + 3 + 3 + 3
COMMAND_SCHEDULE_NAMES = (
    "lin_vel_x_min_schedule",
    "lin_vel_x_max_schedule",
    "lin_vel_y_schedule",
    "ang_vel_yaw_schedule",
    "non_omni_pos_y_schedule",
)
GAIT_FREQUENCY_FEATURE_NAMES = (
    "gait_frequency_min",
    "gait_frequency_max",
    "gait_frequency_lin_vel_ref",
    "gait_frequency_ang_vel_ref",
    "gait_frequency_ang_vel_weight",
)


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def format_float_sequence(values, precision=3):
    eps = 0.5 * (10 ** -precision)
    formatted = []
    for value in values:
        value = float(value)
        if abs(value) < eps:
            value = 0.0
        formatted.append(f"{value:.{precision}f}")
    return "[" + ", ".join(formatted) + "]"


def format_env_vec(tensor, env_id=0, precision=3):
    return format_float_sequence(tensor[env_id].detach().cpu().tolist(), precision=precision)


def observes_goal_height_reference_mask(cfg) -> bool:
    return bool(
        cfg.goal_ee.sphere_center.mixed_height_reference
        or cfg.goal_ee.ranges.ee_goal_sampling_mode in ("body_importance", "body_importance_single_dir")
    )


def compute_num_proprio_from_features(cfg) -> int:
    num_proprio = OBSERVATION_BASE_PROPRIO
    if bool(cfg.env.observe_foot_contacts):
        num_proprio += 4
    if observes_goal_height_reference_mask(cfg):
        num_proprio += 1
    if bool(cfg.env.observe_gait_commands):
        num_proprio += 1 + 4
    return int(num_proprio)


def sync_observation_dims(cfg):
    cfg.action_space = cfg.env.num_actions
    cfg.env.num_observations = cfg.env.num_proprio * (cfg.env.history_len + 1) + cfg.env.num_priv
    cfg.observation_space = cfg.env.num_observations


def parse_float_schedule(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(part) for part in value]

    text = str(value).strip()
    if text == "":
        return None
    if text[0] in "[(":
        parsed = ast.literal_eval(text)
        if not isinstance(parsed, (list, tuple)):
            raise ValueError(f"Expected schedule list/tuple, got {type(parsed).__name__}: {value!r}")
        return [float(part) for part in parsed]
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def build_common_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="b2z1", choices=["b2z1"])
    parser.add_argument("--proj_name", type=str, default="")
    parser.add_argument("--exptid", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="-1")
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--log_root", type=str, default="")
    parser.add_argument("--run_metadata_path", type=str, default="")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--rl_device", type=str, default="cuda:0")
    parser.add_argument("--viewer_display_mode", type=str, default="mesh")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--teleop_mode", action="store_true")
    parser.add_argument("--teleop_input_regularization", action="store_true")
    parser.add_argument("--action_delay_mode", type=str, default=None)
    parser.add_argument("--scene_usd_path", type=str, default=None)
    parser.add_argument("--scene_prim_path", type=str, default=None)
    parser.add_argument("--scene_position", type=float, nargs=3, default=None)
    parser.add_argument("--robot_urdf_path", type=str, default=None)
    parser.add_argument("--base_robot", type=str, default=None, choices=["b2z1"])
    parser.add_argument("--mount_deg", type=float, default=0.0)
    parser.add_argument("--mount_x", type=float, default=None)
    parser.add_argument("--mount_y", type=float, default=None)
    parser.add_argument("--mount_z", type=float, default=None)
    parser.add_argument("--mount_xyz", type=float, nargs=3, default=None)
    parser.add_argument("--robot_collision_profile", type=str, default="full", choices=["full", "feet_gripper"])
    parser.add_argument("--ee_goal_obs_mode", type=str, default=None)
    parser.add_argument(
        "--ee_goal_sampling_mode",
        type=str,
        default=None,
        choices=["arm_front_sphere", "omnidirectional", "body_importance", "body_importance_single_dir"],
    )
    parser.add_argument("--trunk_follow_arm_obs_mode", type=str, default=None)
    parser.add_argument("--trunk_follow_ratio", type=float, default=None)
    parser.add_argument("--mixed_height_reference", type=str_to_bool, nargs="?", const=True, default=None)
    parser.add_argument("--observe_gait_commands", action="store_true")
    parser.add_argument("--observe_foot_contacts", type=str_to_bool, nargs="?", const=True, default=None)
    parser.add_argument("--omnidirectional_pos_y", action="store_true")
    parser.add_argument("--gait_frequency_min", type=float, default=None)
    parser.add_argument("--gait_frequency_max", type=float, default=None)
    parser.add_argument("--episode_length_s", type=float, default=None)
    parser.add_argument("--lin_vel_x_min_schedule", type=str, default=None)
    parser.add_argument("--lin_vel_x_max_schedule", type=str, default=None)
    parser.add_argument("--lin_vel_y_schedule", type=str, default=None)
    parser.add_argument("--ang_vel_yaw_schedule", type=str, default=None)
    parser.add_argument("--non_omni_pos_y_schedule", type=str, default=None)
    parser.add_argument("--reward_scale_preset", type=str, default=None)
    parser.add_argument("--rgb_camera_config", type=str, default=None)
    parser.add_argument("--rgb_camera_backend", type=str, default="auto", choices=["auto", "camera", "tiled"])
    parser.add_argument("--xr_publish_rgb_camera", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--static_default_pose", action="store_true")
    parser.add_argument("--print_timing_breakdown", action="store_true")
    parser.add_argument("--curriculum_iter", type=float, default=None)
    parser.add_argument("--print_force_sensor_every", type=int, default=None)
    parser.add_argument("--wandb_group", type=str, default="")
    return parser


def build_deployment_actor_obs(obs: torch.Tensor, num_proprio, num_priv, history_len) -> torch.Tensor:
    num_proprio = int(num_proprio)
    num_priv = int(num_priv)
    history_len = int(history_len)
    expected_obs_dim = num_proprio * (history_len + 1) + num_priv
    if int(obs.shape[1]) != expected_obs_dim:
        raise ValueError(f"Expected obs dim {expected_obs_dim}, got {int(obs.shape[1])}")
    return torch.cat(
        (
            obs[:, :num_proprio],
            obs[:, num_proprio + num_priv :],
        ),
        dim=1,
    )


def _validate_checkpoint_robot_args(features: dict):
    robot_ablation = str(features.get("robot_ablation") or "none").strip().lower()
    leg_collision_scale = float(features.get("leg_collision_scale", 1.0))
    if robot_ablation not in ("", "none") or leg_collision_scale != 1.0:
        raise ValueError(
            "This trimmed B2Z1 deployment adapter only supports the current low-level checkpoints "
            "with robot_ablation='none' and leg_collision_scale=1.0."
        )


def _apply_checkpoint_goal_and_observation_args(cfg, features: dict):
    cfg.env.ee_goal_obs_mode = str(features["ee_goal_obs_mode"])
    cfg.goal_ee.ranges.ee_goal_sampling_mode = str(features["ee_goal_sampling_mode"])
    cfg.goal_ee.sphere_center.mixed_height_reference = bool(features["mixed_height_reference"])
    cfg.env.observe_gait_commands = bool(features["observe_gait_commands"])
    cfg.env.observe_foot_contacts = bool(features["observe_foot_contacts"])


def _apply_checkpoint_observation_dim_args(cfg, metadata: dict):
    env_metadata = metadata["env_cfg"]["env"]
    cfg.env.num_priv = int(env_metadata["num_priv"])
    cfg.env.history_len = int(env_metadata["history_len"])


def _apply_checkpoint_gait_args(cfg, features: dict, metadata: dict):
    env_metadata = metadata["env_cfg"]["env"]
    for name in GAIT_FREQUENCY_FEATURE_NAMES:
        value = features.get(name, env_metadata.get(name))
        if value is None and name in ("gait_frequency_min", "gait_frequency_max"):
            value = features.get("fixed_trot_frequency", env_metadata.get("fixed_trot_frequency"))
        if value is not None:
            setattr(cfg.env, name, float(value))


def _apply_checkpoint_command_schedule_args(cfg, metadata: dict):
    commands_metadata = metadata["env_cfg"]["commands"]
    for name in COMMAND_SCHEDULE_NAMES:
        parsed = parse_float_schedule(commands_metadata[name])
        if parsed is not None:
            setattr(cfg.commands, name, parsed)


def resolve_mount_xyz(args, features, base_robot):
    if args.mount_xyz is not None:
        return args.mount_xyz

    if features is None:
        default_xyz = list(MOUNT_URDF_SPECS[base_robot]["default_xyz"])
    else:
        default_xyz = [features["mount_x"], features["mount_y"], features["mount_z"]]

    if args.mount_x is not None:
        default_xyz[0] = args.mount_x
    if args.mount_y is not None:
        default_xyz[1] = args.mount_y
    if args.mount_z is not None:
        default_xyz[2] = args.mount_z
    return default_xyz


def resolve_robot_urdf_path(args, require_checkpoint_metadata=False, checkpoint_features=None) -> str:
    if args.robot_urdf_path is not None:
        return args.robot_urdf_path

    base_robot = args.base_robot or args.task
    features = checkpoint_features
    if checkpoint_features is None and (require_checkpoint_metadata or args.ckpt_path or args.exptid):
        try:
            features = run_metadata.load_checkpoint_features(args)
            if len(features) == 0:
                features = None
        except (FileNotFoundError, ValueError):
            features = None

    mount_deg = float(args.mount_deg if features is None else features["mount_deg"])
    mount_xyz = resolve_mount_xyz(args, features, base_robot)
    collision_profile = str(args.robot_collision_profile).strip().lower()
    mount_kwargs = {}
    if collision_profile == "feet_gripper":
        mount_kwargs = {
            "variant_token": "feet_gripper_collision",
            "keep_collision_links": {
                "base_link",
                "FL_foot",
                "FR_foot",
                "RL_foot",
                "RR_foot",
                "gripperStator",
                "gripperMover",
                "gripper_link",
            },
        }
    urdf_rel = ensure_mount_urdf(
        root_dir=str(LOW_LEVEL_ROOT),
        generator_name=base_robot,
        mount_deg=mount_deg,
        mount_xyz=mount_xyz,
        **mount_kwargs,
    )
    urdf_path = str((LOW_LEVEL_ROOT / urdf_rel).resolve())
    print(f"[urdf] auto-generated: {urdf_path}")
    return urdf_path


def _apply_cli_basic_env_args(cfg, args):
    cfg.sim.device = str(args.sim_device)
    cfg.scene.num_envs = int(args.num_envs)
    cfg.env.num_envs = int(args.num_envs)
    cfg.env.teleop_mode = bool(args.teleop_mode)
    cfg.env.teleop_input_regularization = bool(args.teleop_input_regularization)
    if args.action_delay_mode is not None:
        cfg.action_delay_mode = args.action_delay_mode
        cfg.env.action_delay_mode = args.action_delay_mode


def _apply_cli_scene_args(cfg, args):
    if args.scene_usd_path is not None:
        cfg.scene_usd_path = args.scene_usd_path
    if args.scene_prim_path is not None:
        cfg.scene_prim_path = args.scene_prim_path
    if args.scene_position is not None:
        cfg.scene_position = [float(value) for value in args.scene_position]


def _apply_cli_robot_args(cfg, args, for_play: bool, checkpoint_features):
    cfg.robot_urdf_path = resolve_robot_urdf_path(
        args,
        require_checkpoint_metadata=for_play,
        checkpoint_features=checkpoint_features,
    )
    cfg.robot.spawn.asset_path = cfg.robot_urdf_path


def _apply_cli_goal_and_observation_args(cfg, args):
    if args.ee_goal_obs_mode is not None:
        cfg.env.ee_goal_obs_mode = args.ee_goal_obs_mode
    if args.ee_goal_sampling_mode is not None:
        cfg.goal_ee.ranges.ee_goal_sampling_mode = args.ee_goal_sampling_mode
    if args.mixed_height_reference is not None:
        cfg.goal_ee.sphere_center.mixed_height_reference = bool(args.mixed_height_reference)
    if args.observe_gait_commands:
        cfg.env.observe_gait_commands = True
    if args.observe_foot_contacts is not None:
        cfg.env.observe_foot_contacts = bool(args.observe_foot_contacts)
    if args.trunk_follow_arm_obs_mode is not None:
        cfg.env.trunk_follow_arm_obs_mode = args.trunk_follow_arm_obs_mode
    if args.trunk_follow_ratio is not None:
        cfg.goal_ee.sphere_center.trunk_follow_ratio = float(args.trunk_follow_ratio)
    if args.omnidirectional_pos_y:
        cfg.goal_ee.ranges.ee_goal_sampling_mode = "omnidirectional"


def _apply_cli_gait_args(cfg, args):
    if args.gait_frequency_min is not None:
        cfg.env.gait_frequency_min = float(args.gait_frequency_min)
    if args.gait_frequency_max is not None:
        cfg.env.gait_frequency_max = float(args.gait_frequency_max)


def _apply_cli_episode_args(cfg, args):
    if args.episode_length_s is not None:
        cfg.episode_length_s = float(args.episode_length_s)


def _apply_cli_command_schedule_args(cfg, args):
    for name in COMMAND_SCHEDULE_NAMES:
        raw_value = vars(args)[name]
        parsed = parse_float_schedule(raw_value)
        if parsed is not None:
            setattr(cfg.commands, name, parsed)


def _apply_cli_reward_args(cfg, args, apply_reward_scale_preset):
    if args.reward_scale_preset is not None:
        apply_reward_scale_preset(cfg, args.reward_scale_preset)


def _apply_cli_rgb_camera_args(cfg, args):
    rgb_camera_config = isaaclab_app.load_rgb_camera_specs(args.rgb_camera_config)
    cfg.rgb_camera_backend = args.rgb_camera_backend
    if rgb_camera_config is None:
        return

    cfg.rgb_camera_specs = list(rgb_camera_config["cameras"])
    cfg.rgb_camera_draw_in_viewer = bool(rgb_camera_config["draw_in_viewer"])


def _apply_cli_play_args(cfg, args):
    cfg.enable_height_scan = False
    cfg.enable_contact_sensor = False
    cfg.compute_rewards = False
    cfg.profile_env_step = bool(args.print_timing_breakdown)
    cfg.check_terminations = False
    cfg.robot.spawn.activate_contact_sensors = False
    cfg.robot.spawn.articulation_props.enabled_self_collisions = False
    cfg.env.teleop_debug = False
    cfg.commands.curriculum_playback_counter = (
        float(args.curriculum_iter)
        if args.curriculum_iter is not None
        else float(run_metadata.resolved_checkpoint_number(args))
    )


def build_env_cfg(args, for_play=False):
    from legged_gym.envs.manip_loco.b2z1_config import B2Z1IsaacLabCfg, apply_reward_scale_preset

    cfg = B2Z1IsaacLabCfg()
    metadata = run_metadata.load_checkpoint_metadata(args) if for_play else None
    if metadata is None:
        checkpoint_features = None
    else:
        checkpoint_features = metadata["checkpoint_features"]
        _validate_checkpoint_robot_args(checkpoint_features)
        _apply_checkpoint_goal_and_observation_args(cfg, checkpoint_features)
        _apply_checkpoint_observation_dim_args(cfg, metadata)
        _apply_checkpoint_gait_args(cfg, checkpoint_features, metadata)
        _apply_checkpoint_command_schedule_args(cfg, metadata)

    _apply_cli_basic_env_args(cfg, args)
    _apply_cli_scene_args(cfg, args)
    _apply_cli_robot_args(cfg, args, for_play, checkpoint_features)
    _apply_cli_goal_and_observation_args(cfg, args)
    _apply_cli_gait_args(cfg, args)
    _apply_cli_episode_args(cfg, args)
    _apply_cli_command_schedule_args(cfg, args)
    _apply_cli_reward_args(cfg, args, apply_reward_scale_preset)
    _apply_cli_rgb_camera_args(cfg, args)
    if for_play:
        _apply_cli_play_args(cfg, args)

    cfg.env.num_proprio = compute_num_proprio_from_features(cfg)
    sync_observation_dims(cfg)

    if cfg.rgb_camera_specs:
        cfg.sim.render_interval = max(1, int(round(float(cfg.rgb_camera_update_period_s) / float(cfg.sim.dt))))
    else:
        cfg.sim.render_interval = int(cfg.decimation)
    return cfg

def _load_run_metadata_for_checkpoint(ckpt_path: str) -> dict:
    ckpt = Path(ckpt_path).resolve()
    metadata = run_metadata.load_run_metadata(ckpt.parent)
    if metadata is None:
        raise FileNotFoundError(
            f"No run_metadata*.json found next to checkpoint {ckpt}. "
            "Cannot reconstruct low-level policy config from training metadata."
        )
    return metadata


def _metadata_value(metadata: dict, path: tuple[str, ...]):
    value = metadata
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(".".join(path))
        value = value[key]
    return value


def _actor_critic_kwargs_from_metadata(metadata: dict, ckpt: dict) -> dict:
    policy_cfg = _metadata_value(metadata, ("train_cfg", "policy"))
    num_prop = int(_metadata_value(metadata, ("env_cfg", "env", "num_proprio")))
    num_priv = int(_metadata_value(metadata, ("env_cfg", "env", "num_priv")))
    num_hist = int(_metadata_value(metadata, ("env_cfg", "env", "history_len")))
    state_dict = ckpt["model_state_dict"]

    history_weight = state_dict.get("actor.history_encoder.encoder.0.weight")
    actor_weight = state_dict.get("actor.actor_backbone.0.weight")
    critic_weight = state_dict.get("critic.critic_backbone.0.weight")
    if history_weight is None or actor_weight is None or critic_weight is None:
        raise RuntimeError("Checkpoint is missing low-level policy weights required for metadata validation.")

    priv_encoder_dims = list(policy_cfg["priv_encoder_dims"])
    priv_latent_dim = int(priv_encoder_dims[-1]) if priv_encoder_dims else num_priv
    expected = {
        "num_prop": int(history_weight.shape[1]),
        "actor_input": int(actor_weight.shape[1]),
        "critic_input": int(critic_weight.shape[1]),
    }
    actual = {
        "num_prop": num_prop,
        "actor_input": num_prop + priv_latent_dim,
        "critic_input": num_prop + num_priv,
    }
    if expected != actual:
        raise RuntimeError(
            f"Checkpoint metadata from {metadata['_source']} does not match model weights: "
            f"metadata={actual}, checkpoint={expected}"
        )

    return {
        "num_actor_obs": num_prop,
        "num_critic_obs": num_prop,
        "num_actions": int(state_dict["std"].numel()),
        "actor_hidden_dims": list(policy_cfg["actor_hidden_dims"]),
        "critic_hidden_dims": list(policy_cfg["critic_hidden_dims"]),
        "leg_control_head_hidden_dims": list(policy_cfg["leg_control_head_hidden_dims"]),
        "arm_control_head_hidden_dims": list(policy_cfg["arm_control_head_hidden_dims"]),
        "priv_encoder_dims": priv_encoder_dims,
        "activation": policy_cfg["activation"],
        "num_leg_actions": int(policy_cfg["num_leg_actions"]),
        "num_arm_actions": int(policy_cfg["num_arm_actions"]),
        "adaptive_arm_gains": bool(policy_cfg["adaptive_arm_gains"]),
        "adaptive_arm_gains_scale": float(policy_cfg["adaptive_arm_gains_scale"]),
        "num_priv": num_priv,
        "num_hist": num_hist,
        "num_prop": num_prop,
        "output_tanh": bool(policy_cfg["output_tanh"]),
    }


def load_policy_from_checkpoint(ckpt_path: str, device):
    from gr00t.rl.envs.base_task.b2z1_command_base import _B2Z1LowLevelPolicy

    ckpt = torch.load(ckpt_path, map_location=device)
    metadata = _load_run_metadata_for_checkpoint(ckpt_path)
    actor_critic_kwargs = _actor_critic_kwargs_from_metadata(metadata, ckpt)

    policy_cfg = {
        "activation": actor_critic_kwargs["activation"],
        "num_proprio": int(actor_critic_kwargs["num_prop"]),
        "num_priv": int(actor_critic_kwargs["num_priv"]),
        "history_len": int(actor_critic_kwargs["num_hist"]),
        "priv_encoder_dims": list(actor_critic_kwargs["priv_encoder_dims"]),
        "actor_hidden_dims": list(actor_critic_kwargs["actor_hidden_dims"]),
        "leg_control_head_hidden_dims": list(actor_critic_kwargs["leg_control_head_hidden_dims"]),
        "arm_control_head_hidden_dims": list(actor_critic_kwargs["arm_control_head_hidden_dims"]),
        "num_leg_actions": int(actor_critic_kwargs["num_leg_actions"]),
        "num_arm_actions": int(actor_critic_kwargs["num_arm_actions"]),
        "output_tanh": bool(actor_critic_kwargs["output_tanh"]),
    }
    policy = _B2Z1LowLevelPolicy(policy_cfg).to(device)
    incompatible = policy.load_state_dict(ckpt["model_state_dict"], strict=False)
    missing_actor = [key for key in incompatible.missing_keys if key.startswith("actor.")]
    if missing_actor:
        raise RuntimeError(f"Checkpoint is missing actor weights: {missing_actor}")
    policy.eval()
    return policy
