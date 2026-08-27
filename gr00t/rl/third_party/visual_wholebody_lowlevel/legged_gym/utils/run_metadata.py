from __future__ import annotations

import json
import math
import os
from pathlib import Path

from legged_gym.utils.b2z1_mount import normalize_mount_deg, normalize_mount_xyz

RUN_METADATA_FILENAME = "run_metadata.json"
CHECKPOINT_FEATURE_NAMES = (
    "observe_gait_commands",
    "observe_foot_contacts",
    "mixed_height_reference",
    "ee_goal_sampling_mode",
    "ee_goal_obs_mode",
    "reward_scale_preset",
    "gait_frequency_min",
    "gait_frequency_max",
    "gait_frequency_lin_vel_ref",
    "gait_frequency_ang_vel_ref",
    "gait_frequency_ang_vel_weight",
    "mount_deg",
    "mount_x",
    "mount_y",
    "mount_z",
)


def get_log_root(args) -> Path:
    from legged_gym.envs.manip_loco.b2z1_config import B2Z1IsaacLabCfg

    cfg = B2Z1IsaacLabCfg()
    return Path(args.log_root or os.environ.get("LEGGED_GYM_LOG_ROOT", cfg.paths.log_root))


def get_run_log_dir(args) -> Path:
    proj_name = args.proj_name or f"{args.task}-low"
    if not args.exptid:
        raise ValueError("--exptid is required when --ckpt_path is not set.")
    return get_log_root(args) / proj_name / args.exptid


def get_run_metadata_filename(filename=None):
    if filename is not None:
        return filename
    return os.environ.get("LEGGED_GYM_RUN_METADATA_FILENAME", RUN_METADATA_FILENAME)


def get_run_metadata_path(log_dir, filename=None) -> Path:
    return Path(log_dir) / get_run_metadata_filename(filename)


def load_run_metadata(log_dir, filename=None):
    log_dir = Path(log_dir)
    preferred_path = get_run_metadata_path(log_dir, filename=filename)
    metadata_candidates = []
    if preferred_path.is_file():
        metadata_candidates.append(preferred_path)

    extra_candidates = [
        path for path in log_dir.glob("run_metadata*.json")
        if path.is_file() and path not in metadata_candidates
    ]
    extra_candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    metadata_candidates.extend(extra_candidates)

    if not metadata_candidates:
        return None

    metadata_path = metadata_candidates[0]
    with metadata_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data["_source"] = str(metadata_path)
    return data


def checkpoint_model_path(args) -> Path:
    if args.ckpt_path:
        return Path(args.ckpt_path).resolve()
    run_dir = get_run_log_dir(args)
    ckpt = str(args.checkpoint or "-1")
    if ckpt == "-1":
        candidates = []
        for path in run_dir.glob("model_*.pt"):
            stem = path.stem.replace("model_", "", 1)
            try:
                candidates.append((int(stem), path))
            except ValueError:
                continue
        if not candidates:
            raise FileNotFoundError(f"No model_*.pt checkpoints found in {run_dir}")
        return max(candidates, key=lambda item: item[0])[1].resolve()
    return (run_dir / f"model_{ckpt}.pt").resolve()


def checkpoint_number_from_path(path) -> int:
    stem = Path(path).stem
    if stem.startswith("model_"):
        stem = stem.replace("model_", "", 1)
    return int(stem)


def resolved_checkpoint_number(args) -> int:
    ckpt = str(args.checkpoint or "-1")
    if ckpt != "-1":
        return int(ckpt)
    return checkpoint_number_from_path(checkpoint_model_path(args))


def load_checkpoint_metadata(args) -> dict | None:
    if args.run_metadata_path:
        metadata_path = Path(args.run_metadata_path)
        if not metadata_path.exists():
            print(f"[metadata] run metadata not found: {metadata_path}; use CLI args/defaults.")
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[metadata] failed to parse run metadata {metadata_path}: {exc}; use CLI args/defaults.")
            return None
        metadata["_source"] = str(metadata_path)
    else:
        ckpt_path = checkpoint_model_path(args)
        metadata = load_run_metadata(ckpt_path.parent)
    if metadata is None:
        print(f"[metadata] no run_metadata*.json found; use CLI args/defaults.")
        return None
    return metadata


def load_checkpoint_features(args) -> dict:
    metadata = load_checkpoint_metadata(args)
    if metadata is None:
        return {}
    features = metadata["checkpoint_features"]
    print(f"[metadata] loaded checkpoint_features from: {metadata['_source']}")
    return features if isinstance(features, dict) else {}


def _extract_checkpoint_features(args, env_cfg):
    if env_cfg is None:
        raise ValueError("env_cfg is required when extracting checkpoint features")
    mount_xyz = normalize_mount_xyz(env_cfg.goal_ee.urdf_mount.arm_base_offset)
    mixed_height_reference = env_cfg.goal_ee.sphere_center.mixed_height_reference
    return {
        "observe_gait_commands": bool(env_cfg.env.observe_gait_commands),
        "observe_foot_contacts": bool(env_cfg.env.observe_foot_contacts),
        "mixed_height_reference": bool(mixed_height_reference),
        "ee_goal_sampling_mode": str(env_cfg.goal_ee.ranges.ee_goal_sampling_mode),
        "ee_goal_obs_mode": str(env_cfg.env.ee_goal_obs_mode),
        "reward_scale_preset": str(env_cfg.rewards.reward_scale_preset),
        "gait_frequency_min": float(env_cfg.env.gait_frequency_min),
        "gait_frequency_max": float(env_cfg.env.gait_frequency_max),
        "gait_frequency_lin_vel_ref": float(env_cfg.env.gait_frequency_lin_vel_ref),
        "gait_frequency_ang_vel_ref": float(env_cfg.env.gait_frequency_ang_vel_ref),
        "gait_frequency_ang_vel_weight": float(env_cfg.env.gait_frequency_ang_vel_weight),
        "mount_deg": normalize_mount_deg(math.degrees(float(env_cfg.goal_ee.urdf_mount.mount_yaw_offset))),
        "mount_x": mount_xyz[0],
        "mount_y": mount_xyz[1],
        "mount_z": mount_xyz[2],
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _class_to_dict(obj):
    if isinstance(obj, dict):
        return {str(k): _class_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_class_to_dict(v) for v in obj]
    if not hasattr(obj, "__dict__") and not isinstance(obj, type):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        value = getattr(obj, key)
        if callable(value):
            continue
        result[key] = _class_to_dict(value)
    return result


def write_run_metadata(log_dir: Path, args, env_cfg, train_cfg=None):
    log_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "checkpoint_features": _extract_checkpoint_features(args, env_cfg),
        "cli_args": _json_safe(vars(args)),
        "env": {
            "robot_usd_path": env_cfg.robot_usd_path,
            "num_envs": env_cfg.scene.num_envs,
        },
        "env_cfg": _json_safe(_class_to_dict(env_cfg)),
    }
    if train_cfg is not None:
        metadata["train_cfg"] = _json_safe(_class_to_dict(train_cfg))
    metadata_path = get_run_metadata_path(log_dir)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata_path


def update_run_metadata(log_dir: Path, updates: dict, filename=None):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_run_metadata(log_dir, filename=filename) or {}
    metadata.pop("_source", None)
    for key, value in updates.items():
        metadata[key] = _json_safe(value)
    metadata_path = get_run_metadata_path(log_dir, filename=filename)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata_path
