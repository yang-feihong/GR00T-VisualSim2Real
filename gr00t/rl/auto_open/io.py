"""AutoOpen provenance snapshots and RGB artifact writers."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_ROOT = Path(__file__).resolve().parents[1]


def standard_simulation_provenance(args, runner, base_seed):
    """Build reproducibility metadata without changing simulator state."""
    from gr00t.rl.data_collection.auto_open_writer import file_sha256

    repo = SCRIPT_ROOT.parent.parent

    def git(*git_args):
        try:
            return subprocess.check_output(
                ("git", *git_args), cwd=repo, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    def hashed(path):
        path = Path(path)
        return {"path": str(path), "sha256": file_sha256(path)} if path.is_file() else None

    try:
        import omni.kit.app

        isaac_sim_version = str(omni.kit.app.get_app().get_app_version())
    except Exception:
        isaac_sim_version = "unknown"
    try:
        driver_version = subprocess.check_output(
            ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        driver_version = "unknown"

    robot_spawn = runner.env.cfg.robot.spawn
    robot_props = robot_spawn.articulation_props
    collision_props = robot_spawn.collision_props
    rigid_props = robot_spawn.rigid_props
    sim_cfg = runner.env.cfg.sim
    physx_cfg = sim_cfg.physx
    camera_path = Path(args.rgb_camera_config)
    nominal_camera_path = Path(args.nominal_rgb_camera_config)
    door_source = repo / "gr00t" / "rl" / "isaac_utils" / "playground" / "env_rand" / "door.py"
    camera_configuration = json.loads(camera_path.read_text(encoding="utf-8"))
    return {
        "collector": "auto_open_standard_sidecar",
        "repository_commit": git("rev-parse", "HEAD"),
        "repository_dirty": bool(git("status", "--porcelain")),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "nvidia_driver_version": driver_version,
        "isaac_sim_version": isaac_sim_version,
        "scene_id": "door_open_b2z1_autoopen",
        "robot_asset": hashed(args.robot_usd_path),
        "door_asset": {
            "kind": "procedural_usd_stage",
            "generator_source": hashed(door_source),
            "episode_stage_snapshots": "stored in provenance.episodes.*.door_asset",
        },
        "checkpoint": hashed(args.ckpt_path),
        "run_metadata": hashed(args.run_metadata_path),
        "camera_config": hashed(camera_path),
        "nominal_camera_config": hashed(nominal_camera_path),
        "camera_calibration_id": file_sha256(camera_path),
        "camera_mount_randomization": args.camera_mount_randomization,
        "camera_configuration": camera_configuration,
        "camera_rotation_convention": {
            "input": "roll_pitch_yaw_radians_relative_to_parent_body",
            "composition": "q = qz(yaw) * qy(pitch) * qx(roll)",
            "runtime_quaternion_order": "wxyz",
            "dataset_pose_quaternion_order": "xyzw",
        },
        "camera_rendering": {
            "renderer": "RTX",
            "color_output": "RGB_uint8_sRGB",
            "exposure": "Isaac Sim renderer default; no per-camera override",
            "reflections": bool(args.rtx_reflections),
            "translucency": bool(args.rtx_translucency),
            "subsurface_scattering": bool(args.rtx_subsurface_scattering),
            "caustics": bool(args.rtx_caustics),
            "indirect_diffuse": bool(args.rtx_indirect_diffuse),
            "ambient_occlusion": bool(args.rtx_ambient_occlusion),
            "direct_lighting_samples_per_pixel": int(args.rtx_direct_lighting_spp),
            "reflection_samples_per_pixel": int(args.rtx_reflections_spp),
        },
        "physics_dt_s": float(runner.env.physics_dt),
        "control_dt_s": float(runner.env.dt),
        "decimation": int(runner.env.cfg.decimation),
        "solver_position_iterations": int(
            getattr(robot_props, "solver_position_iteration_count", 0)
        ),
        "solver_velocity_iterations": int(
            getattr(robot_props, "solver_velocity_iteration_count", 0)
        ),
        "ccd_enabled": bool(getattr(robot_props, "enable_ccd", False)),
        "self_collisions_enabled": bool(robot_props.enabled_self_collisions),
        "gravity_m_s2": list(sim_cfg.gravity),
        "physx_scene": {
            name: getattr(physx_cfg, name, None)
            for name in (
                "solver_type",
                "max_position_iteration_count",
                "max_velocity_iteration_count",
                "bounce_threshold_velocity",
                "friction_offset_threshold",
                "friction_correlation_distance",
                "enable_stabilization",
                "enable_enhanced_determinism",
                "gpu_max_rigid_contact_count",
                "gpu_max_rigid_patch_count",
                "gpu_found_lost_pairs_capacity",
                "gpu_found_lost_aggregate_pairs_capacity",
                "gpu_total_aggregate_pairs_capacity",
                "gpu_collision_stack_size",
                "gpu_heap_capacity",
                "gpu_temp_buffer_capacity",
            )
        },
        "robot_rigid_body": {
            name: getattr(rigid_props, name, None)
            for name in (
                "disable_gravity",
                "max_linear_velocity",
                "max_angular_velocity",
                "max_depenetration_velocity",
                "enable_gyroscopic_forces",
            )
        },
        "robot_collision": {
            "contact_offset_m": getattr(collision_props, "contact_offset", None),
            "rest_offset_m": getattr(collision_props, "rest_offset", None),
            "source": "robot_asset collider subtree",
        },
        "seed_start": int(base_seed),
        "randomization": "DoorSpawner metadata recorded per episode",
        "control_backend": "B2Z1 low-level policy plus AutoOpen FSM",
        "rendering": "RTX headless, two DATA_STANARD.md RGB sensors at 50 Hz",
    }


def usd_json_value(value):
    """Convert authored USD values to stable JSON without changing the stage."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "path"):
        return str(value.path)
    if hasattr(value, "GetReal") and hasattr(value, "GetImaginary"):
        imaginary = value.GetImaginary()
        return {
            "real": float(value.GetReal()),
            "imaginary": [float(component) for component in imaginary],
        }
    try:
        return [usd_json_value(item) for item in value]
    except TypeError:
        return str(value)


def door_asset_stage_snapshot(stage, door_prim):
    """Describe and hash the actual generated door subtree for one episode."""
    from pxr import Usd

    prims = []
    for prim in Usd.PrimRange(door_prim, Usd.TraverseInstanceProxies()):
        attributes = {}
        for attribute in prim.GetAttributes():
            if attribute.HasAuthoredValueOpinion():
                attributes[attribute.GetName()] = usd_json_value(attribute.Get())
        relationships = {}
        for relationship in prim.GetRelationships():
            targets = relationship.GetTargets()
            if targets:
                relationships[relationship.GetName()] = [str(path) for path in targets]
        prims.append(
            {
                "path": str(prim.GetPath()),
                "type": prim.GetTypeName(),
                "attributes": attributes,
                "relationships": relationships,
                "custom_data": usd_json_value(prim.GetCustomData()),
            }
        )
    payload = {
        "stage_prim_path": str(door_prim.GetPath()),
        "prims": prims,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    payload["sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def rgb_array(frame):
    if frame is None:
        return None
    array = frame.detach().cpu().numpy() if torch.is_tensor(frame) else np.asarray(frame)
    if array.ndim == 4:
        array = array[0]
    if array.shape[-1] == 4:
        array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(
            array * (255.0 if array.max(initial=0.0) <= 1.0 else 1.0), 0, 255
        ).astype(np.uint8)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def save_rgb_video(frames, timestamps, path: Path):
    if not frames:
        return False
    import imageio.v2 as imageio

    if len(timestamps) >= 2:
        intervals = np.diff(np.asarray(timestamps, dtype=np.float64))
        intervals = intervals[intervals > 0.0]
        fps = 1.0 / float(np.median(intervals)) if len(intervals) else 4.0
    else:
        fps = 4.0
    writer = imageio.get_writer(
        str(path), fps=fps, codec="libx264", quality=8, macro_block_size=1
    )
    try:
        for frame in frames:
            writer.append_data(rgb_array(frame))
    finally:
        writer.close()
    return True


def save_rgb_mosaic_video(camera_frames, camera_timestamps, path: Path):
    preferred_order = (
        "wrist",
        "trunk_front",
        "robot_top_diagnostic",
        "door_world_diagnostic",
        "door_panel_diagnostic",
        "door_panel_top_diagnostic",
    )
    camera_names = [name for name in preferred_order if camera_frames.get(name)]
    camera_names.extend(
        sorted(
            name
            for name, frames in camera_frames.items()
            if frames and name not in camera_names
        )
    )
    if not camera_names:
        return False

    reference_name = max(camera_names, key=lambda name: len(camera_frames[name]))
    timeline = np.asarray(camera_timestamps[reference_name], dtype=np.float64)
    if timeline.size == 0:
        return False
    if timeline.size >= 2:
        intervals = np.diff(timeline)
        intervals = intervals[intervals > 0.0]
        fps = 1.0 / float(np.median(intervals)) if len(intervals) else 4.0
    else:
        fps = 4.0

    from PIL import Image, ImageDraw
    import imageio.v2 as imageio

    tile_width, tile_height = 480, 360
    canvas_width, canvas_height = 3 * tile_width, 2 * tile_height
    timestamp_arrays = {
        name: np.asarray(camera_timestamps[name], dtype=np.float64)
        for name in camera_names
    }

    def nearest_frame(name, timestamp):
        timestamps = timestamp_arrays[name]
        index = int(np.searchsorted(timestamps, timestamp, side="left"))
        if index >= len(timestamps):
            index = len(timestamps) - 1
        elif (
            index > 0
            and timestamp - timestamps[index - 1]
            <= timestamps[index] - timestamp
        ):
            index -= 1
        return camera_frames[name][index]

    writer = imageio.get_writer(
        str(path), fps=fps, codec="libx264", quality=8, macro_block_size=1
    )
    try:
        for timestamp in timeline:
            canvas = Image.new("RGB", (canvas_width, canvas_height), color=(0, 0, 0))
            for tile_index, camera_name in enumerate(camera_names[:6]):
                frame = rgb_array(nearest_frame(camera_name, timestamp))
                tile = Image.fromarray(frame, mode="RGB")
                if tile.size != (tile_width, tile_height):
                    tile = tile.resize(
                        (tile_width, tile_height), Image.Resampling.BILINEAR
                    )
                draw = ImageDraw.Draw(tile)
                label_box = draw.textbbox((0, 0), camera_name)
                label_width = label_box[2] - label_box[0]
                label_height = label_box[3] - label_box[1]
                draw.rectangle(
                    (8, 8, 20 + label_width, 18 + label_height), fill=(0, 0, 0)
                )
                draw.text((14, 12), camera_name, fill=(255, 255, 255))
                x = (tile_index % 3) * tile_width
                y = (tile_index // 3) * tile_height
                canvas.paste(tile, (x, y))
            writer.append_data(np.asarray(canvas, dtype=np.uint8))
    finally:
        writer.close()
    return True
