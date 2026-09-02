import json
import os
import json
import time

import numpy as np


REPRESENTATIVE_RGB_CAMERA_BODY_HINTS = {
    "base_link": "B2 trunk/base representative. Use this for fixed front sensors on the dog body.",
    "trunk": "B1 trunk/base representative. Use this for fixed front sensors on the dog body.",
    "link00": "Z1 arm base / shoulder mount representative.",
    "link03": "Z1 elbow/forearm moving segment.",
    "link05": "Z1 wrist forearm-roll moving segment.",
    "link06": "Z1 wrist/end moving segment. Default wrist stereo mount.",
    "gripper_link": "B2Z1 gripper / end-effector representative.",
    "ee_gripper_link": "B1Z1 gripper / end-effector representative.",
    "FL_foot": "Front-left foot representative.",
    "FR_foot": "Front-right foot representative.",
    "RL_foot": "Rear-left foot representative.",
    "RR_foot": "Rear-right foot representative.",
}


def parse_optional_vec3(value, name):
    if value is None or str(value).strip() == "":
        return None
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 3:
        raise ValueError(f"{name} must contain exactly three comma-separated values.")
    return tuple(float(part) for part in parts)


def parse_vec3(value, name):
    if isinstance(value, str):
        parsed = parse_optional_vec3(value, name)
    else:
        parsed = value
    if parsed is None or len(parsed) != 3:
        raise ValueError(f"{name} must contain exactly three values.")
    return tuple(float(part) for part in parsed)


def load_rgb_camera_config(path):
    path = str(path or "").strip()
    if not path:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"RGB camera config not found: {path}")

    suffix = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as config_file:
        if suffix in (".yaml", ".yml"):
            try:
                import yaml
            except Exception as exc:
                raise RuntimeError(
                    "YAML camera configs require PyYAML. Use JSON or install PyYAML."
                ) from exc
            return yaml.safe_load(config_file)
        return json.load(config_file)


def require_config_value(config, key, context):
    if key not in config:
        raise ValueError(f"{context} must set '{key}'.")
    return config[key]


def rgb_camera_specs_from_config(config):
    if not isinstance(config, dict):
        raise ValueError("RGB camera config must be a JSON/YAML object.")
    if not config or not bool(require_config_value(config, "enabled", "RGB camera config")):
        return {
            "enabled": False,
            "draw_in_viewer": False,
            "preview_window": False,
            "preview_scale": 1.0,
            "preview_fps": 0.0,
            "cameras": [],
        }

    defaults = dict(require_config_value(config, "defaults", "RGB camera config"))
    preview_scale = float(require_config_value(config, "preview_scale", "RGB camera config"))
    if preview_scale <= 0.0:
        raise ValueError("RGB camera config preview_scale must be positive.")

    camera_configs = require_config_value(config, "cameras", "RGB camera config")
    if not isinstance(camera_configs, list) or len(camera_configs) == 0:
        raise ValueError("RGB camera config must contain a non-empty 'cameras' list.")

    diagnostic_configs = {}
    for section_name, section in config.items():
        if not isinstance(section, dict) or "camera_name" not in section:
            continue
        camera_name = str(section["camera_name"]).strip()
        if not camera_name:
            raise ValueError(f"RGB camera section '{section_name}' has an empty camera_name.")
        required_keys = (
            "target_reference",
            "target_offset_leaf_m",
            "view_axis_reference",
            "view_axis_local",
            "view_axis_direction_sign",
            "axis_distance_m",
            "fill_light_up_offset_m",
            "fill_light_right_offset_m",
            "fill_light_intensity",
            "fill_light_radius_m",
        )
        if any(key not in section for key in required_keys):
            continue
        if camera_name in diagnostic_configs:
            raise ValueError(f"Duplicate door diagnostic config for camera: {camera_name}")
        target_reference = str(section["target_reference"])
        view_axis_reference = str(section["view_axis_reference"])
        if target_reference != "handle_joint":
            raise ValueError(
                f"Door diagnostic camera '{camera_name}' requires target_reference='handle_joint'."
            )
        if view_axis_reference not in ("handle_body", "door_leaf"):
            raise ValueError(
                f"Door diagnostic camera '{camera_name}' has unsupported "
                f"view_axis_reference={view_axis_reference!r}."
            )
        axis_distance = float(section["axis_distance_m"])
        if axis_distance <= 0.0:
            raise ValueError(f"Door diagnostic camera '{camera_name}' axis_distance_m must be positive.")
        diagnostic_configs[camera_name] = {
            **section,
            "camera_name": camera_name,
            "target_offset_leaf_m": parse_vec3(
                section["target_offset_leaf_m"], f"{section_name}.target_offset_leaf_m"
            ),
            "view_axis_local": parse_vec3(
                section["view_axis_local"], f"{section_name}.view_axis_local"
            ),
            "axis_distance_m": axis_distance,
        }

    specs = []
    names = set()
    for index, camera_cfg in enumerate(camera_configs):
        if not isinstance(camera_cfg, dict):
            raise ValueError(f"RGB camera config #{index} must be a JSON/YAML object.")
        if not bool(require_config_value(camera_cfg, "enabled", f"RGB camera config #{index}")):
            continue

        camera_name = str(camera_cfg["name"]).strip()
        if not camera_name:
            raise ValueError(f"Camera config #{index} has an empty name.")
        if camera_name in names:
            raise ValueError(f"Duplicate RGB camera name: {camera_name}")
        names.add(camera_name)

        body_name = str(require_config_value(camera_cfg, "body_name", f"Camera '{camera_name}'")).strip()
        if not body_name:
            raise ValueError(f"Camera '{camera_name}' has an empty body_name.")

        camera_mode = str(camera_cfg.get("mode", require_config_value(defaults, "mode", "RGB camera defaults"))).strip().lower()
        if camera_mode not in ("mono", "stereo"):
            raise ValueError(f"Camera '{camera_name}' has unsupported mode: {camera_mode}")

        mono_offset = parse_vec3(
            camera_cfg.get("mono_offset", require_config_value(defaults, "mono_offset", "RGB camera defaults")),
            f"{camera_name}.mono_offset",
        )
        left_offset = parse_vec3(
            camera_cfg.get("left_offset", require_config_value(defaults, "left_offset", "RGB camera defaults")),
            f"{camera_name}.left_offset",
        )
        right_offset = parse_vec3(
            camera_cfg.get("right_offset", require_config_value(defaults, "right_offset", "RGB camera defaults")),
            f"{camera_name}.right_offset",
        )
        if "position" not in camera_cfg and "local_position" not in camera_cfg:
            raise ValueError(f"Camera '{camera_name}' must set position/local_position.")
        if "rotation" not in camera_cfg and "local_rotation" not in camera_cfg:
            raise ValueError(f"Camera '{camera_name}' must set rotation/local_rotation.")

        projection_type = str(camera_cfg.get("projection_type", defaults.get("projection_type", "pinhole"))).strip()
        supported_projection_types = {
            "pinhole",
            "opencvFisheye",
            "fisheyePolynomial",
            "fisheyeSpherical",
            "fisheyeKannalaBrandtK3",
            "fisheyeRadTanThinPrism",
            "omniDirectionalStereo",
        }
        if projection_type not in supported_projection_types:
            raise ValueError(
                f"Camera '{camera_name}' has unsupported projection_type={projection_type!r}; "
                f"expected one of {sorted(supported_projection_types)}."
            )

        width = int(camera_cfg.get("width", require_config_value(defaults, "width", "RGB camera defaults")))
        height = int(camera_cfg.get("height", require_config_value(defaults, "height", "RGB camera defaults")))
        if width <= 0 or height <= 0:
            raise ValueError(f"Camera '{camera_name}' width and height must be positive.")

        valid_circle_cfg = camera_cfg.get("valid_circle_mask", defaults.get("valid_circle_mask"))
        valid_circle_mask = None
        if valid_circle_cfg is not None and bool(valid_circle_cfg.get("enabled", True)):
            center = valid_circle_cfg.get("center", (0.5 * width, 0.5 * height))
            if not isinstance(center, (list, tuple)) or len(center) != 2:
                raise ValueError(f"Camera '{camera_name}' valid_circle_mask.center must contain two values.")
            radius = float(require_config_value(valid_circle_cfg, "radius", f"Camera '{camera_name}' valid_circle_mask"))
            feather = float(valid_circle_cfg.get("feather", 2.0))
            if radius <= 0.0 or feather < 0.0 or feather > radius:
                raise ValueError(
                    f"Camera '{camera_name}' valid_circle_mask requires radius > 0 and 0 <= feather <= radius."
                )
            valid_circle_mask = {
                "center": (float(center[0]), float(center[1])),
                "radius": radius,
                "feather": feather,
            }

        lens_spec = {"projection_type": projection_type}
        if projection_type == "pinhole":
            lens_spec["horizontal_fov"] = float(
                camera_cfg.get(
                    "horizontal_fov",
                    camera_cfg.get("fov", require_config_value(defaults, "horizontal_fov", "RGB camera defaults")),
                )
            )
        elif projection_type == "opencvFisheye":
            for key in ("fx", "fy", "cx", "cy"):
                lens_spec[key] = float(camera_cfg.get(key, require_config_value(defaults, key, "RGB camera defaults")))
            lens_spec["skew"] = float(camera_cfg.get("skew", defaults.get("skew", 0.0)))
            coefficients = camera_cfg.get(
                "distortion_coefficients",
                require_config_value(defaults, "distortion_coefficients", "RGB camera defaults"),
            )
            if not isinstance(coefficients, (list, tuple)) or len(coefficients) != 4:
                raise ValueError(
                    f"Camera '{camera_name}' distortion_coefficients must contain exactly four values "
                    "for projection_type='opencvFisheye'."
                )
            lens_spec["distortion_coefficients"] = tuple(float(value) for value in coefficients)
        else:
            fisheye_keys = (
                "fisheye_nominal_width",
                "fisheye_nominal_height",
                "fisheye_optical_centre_x",
                "fisheye_optical_centre_y",
                "focal_length_mm",
                "fisheye_max_fov",
                "fisheye_polynomial_a",
                "fisheye_polynomial_b",
                "fisheye_polynomial_c",
                "fisheye_polynomial_d",
                "fisheye_polynomial_e",
                "fisheye_polynomial_f",
            )
            for key in fisheye_keys:
                lens_spec[key] = float(camera_cfg.get(key, require_config_value(defaults, key, "RGB camera defaults")))

        spec = {
                "name": camera_name,
                "mode": camera_mode,
                "body_name": body_name,
                "topic_prefix": str(require_config_value(camera_cfg, "topic_prefix", f"Camera '{camera_name}'")).rstrip("/"),
                "width": width,
                "height": height,
                "fps": float(camera_cfg.get("fps", require_config_value(defaults, "fps", "RGB camera defaults"))),
                "mono_offset": mono_offset,
                "left_offset": left_offset,
                "right_offset": right_offset,
                "local_position": parse_vec3(
                    camera_cfg.get("position", camera_cfg.get("local_position")),
                    f"{camera_name}.position",
                ),
                "local_rotation": parse_vec3(
                    camera_cfg.get("rotation", camera_cfg.get("local_rotation")),
                    f"{camera_name}.rotation",
                ),
                "valid_circle_mask": valid_circle_mask,
                **lens_spec,
            }
        if camera_name in diagnostic_configs:
            spec["door_diagnostic"] = diagnostic_configs[camera_name]
        specs.append(spec)

    unknown_diagnostics = sorted(set(diagnostic_configs) - names)
    if unknown_diagnostics:
        raise ValueError(
            "Door diagnostic configs reference cameras that are not enabled: "
            + ", ".join(unknown_diagnostics)
        )

    return {
        "enabled": True,
        "draw_in_viewer": bool(require_config_value(config, "draw_in_viewer", "RGB camera config")),
        "preview_window": bool(require_config_value(config, "preview_window", "RGB camera config")),
        "preview_scale": preview_scale,
        "preview_fps": float(require_config_value(config, "preview_fps", "RGB camera config")),
        "door_diagnostics": diagnostic_configs,
        "cameras": specs,
    }


class SimRgbCameraManager:
    def __init__(self, env, config_path):
        self.enabled = False
        self.env = env
        self.rigs = []
        self.preview_enabled = False
        self.preview_scale = None
        self.preview_period = 0.0
        self.preview_last_wall_time = 0.0
        self.publish_time_s = 0.0
        self.preview_window_name = "RGB Cameras"
        self._cv2 = None

        config = load_rgb_camera_config(config_path)
        if config is None:
            return
        if not all(hasattr(self.env, name) for name in ("render_stereo_rgb_camera_rigs", "enable_stereo_rgb_camera_debug_draw")):
            print("[rgb_camera][warn] RGB camera config ignored: IsaacLab RGB camera backend is not available yet.")
            return
        self._create_rigs_from_config(config)

    def _create_rigs_from_config(self, config):
        parsed = rgb_camera_specs_from_config(config)
        if not parsed["enabled"]:
            return

        draw_in_viewer = bool(parsed["draw_in_viewer"])
        self.preview_enabled = bool(parsed["preview_window"])
        self.preview_scale = float(parsed["preview_scale"])
        preview_fps = float(parsed["preview_fps"])
        self.preview_period = 1.0 / preview_fps if preview_fps > 0.0 else 0.0

        env_rigs = self.env._rgb_camera_rigs
        for spec in parsed["cameras"]:
            camera_name = spec["name"]
            camera_mode = spec["mode"]
            if camera_name not in env_rigs:
                print(f"[rgb_camera][warn] configured camera {camera_name!r} was not created by the environment.")
                continue
            fps = float(spec["fps"])
            period = 1.0 / fps if fps > 0.0 else 0.0
            camera_update_period = float(self.env.cfg.rgb_camera_update_period_s)
            if camera_update_period > 0.0:
                period = max(period, camera_update_period)
            self.rigs.append(
                {
                    "name": camera_name,
                    "mode": camera_mode,
                    "topic_prefix": spec["topic_prefix"],
                    "publish_period": period,
                    "last_publish_wall_time": 0.0,
                    "last_publish_sim_time": -float("inf"),
                }
            )
            topic = f"{spec['topic_prefix']}/image_raw" if camera_mode == "mono" else f"{spec['topic_prefix']}/{{left,right}}/image_raw"
            print(
                f"[rgb_camera] {camera_name}: mode={camera_mode}, topic={topic}, fps={fps:.2f}"
            )

        if len(self.rigs) == 0:
            print("[rgb_camera] camera config loaded, but no cameras are enabled.")
            return

        self.enabled = True
        if draw_in_viewer:
            self.env.enable_stereo_rgb_camera_debug_draw(True)
        if self.preview_enabled:
            self._open_preview_window()

    def _open_preview_window(self):
        has_display = os.name == "nt" or os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        if not has_display:
            self.preview_enabled = False
            print("[rgb_camera] preview disabled: no DISPLAY or WAYLAND_DISPLAY is set")
            return
        try:
            import cv2
            self._cv2 = cv2
            cv2.namedWindow(self.preview_window_name, cv2.WINDOW_NORMAL)
            print("[rgb_camera] preview window enabled")
        except Exception as exc:
            self.preview_enabled = False
            print(f"[rgb_camera] preview disabled: {exc}")

    def render_images(self, env_id=0, rig_names=None, update=False):
        if not self.enabled:
            return {}
        return self.env.render_stereo_rgb_camera_rigs(
            env_id=env_id,
            rig_names=rig_names,
            update=update,
        )

    def due_rigs(self, now_wall):
        return [
            rig for rig in self.rigs
            if rig["publish_period"] <= 0.0
            or now_wall - rig["last_publish_wall_time"] >= rig["publish_period"]
        ]

    def mark_published(self, rig, now_wall):
        rig["last_publish_wall_time"] = now_wall

    def advance_publish_time(self):
        self.publish_time_s += float(self.env.dt)
        return self.publish_time_s

    def due_rigs_at_sim_time(self, sim_time):
        return [
            rig for rig in self.rigs
            if rig["publish_period"] <= 0.0
            or float(sim_time) - rig["last_publish_sim_time"] >= rig["publish_period"]
        ]

    def mark_published_at_sim_time(self, rig, sim_time):
        rig["last_publish_sim_time"] = float(sim_time)
        rig["last_publish_wall_time"] = time.monotonic()

    def _make_preview_tile(self, rgb, label):
        cv2 = self._cv2
        tile = np.ascontiguousarray(rgb, dtype=np.uint8)
        if self.preview_scale != 1.0:
            width = max(1, int(tile.shape[1] * self.preview_scale))
            height = max(1, int(tile.shape[0] * self.preview_scale))
            tile = cv2.resize(tile, (width, height), interpolation=cv2.INTER_AREA)

        label_h = 26
        preview = np.zeros((tile.shape[0] + label_h, tile.shape[1], 3), dtype=np.uint8)
        preview[label_h:, :, :] = tile
        cv2.putText(
            preview,
            label,
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return preview[:, :, ::-1]

    def show_preview_window(self, images_by_name):
        if not self.preview_enabled or self._cv2 is None:
            return

        rows = []
        for rig in self.rigs:
            images = images_by_name.get(rig["name"])
            if images is None:
                continue
            if rig["mode"] == "mono":
                rows.append(self._make_preview_tile(images[0], rig["name"]))
            else:
                left_rgb, right_rgb = images
                left_tile = self._make_preview_tile(left_rgb, f"{rig['name']} / left")
                right_tile = self._make_preview_tile(right_rgb, f"{rig['name']} / right")
                rows.append(np.concatenate([left_tile, right_tile], axis=1))
        if len(rows) == 0:
            return

        max_width = max(row.shape[1] for row in rows)
        padded_rows = []
        for row in rows:
            if row.shape[1] < max_width:
                pad = np.zeros((row.shape[0], max_width - row.shape[1], 3), dtype=np.uint8)
                row = np.concatenate([row, pad], axis=1)
            padded_rows.append(row)
        mosaic = np.concatenate(padded_rows, axis=0)

        try:
            self._cv2.imshow(self.preview_window_name, mosaic)
            self._cv2.waitKey(1)
        except Exception as exc:
            self.preview_enabled = False
            print(f"[rgb_camera] preview disabled after imshow failure: {exc}")

    def maybe_update_preview(self, env_id=0):
        if not self.enabled or not self.preview_enabled:
            return {}
        now_wall = time.monotonic()
        if self.preview_period > 0.0 and now_wall - self.preview_last_wall_time < self.preview_period:
            return {}
        images_by_name = self.render_images(env_id=env_id)
        self.show_preview_window(images_by_name)
        self.preview_last_wall_time = now_wall
        return images_by_name
