from __future__ import annotations

import math
import time

import torch


class IsaacLabViewerController:
    def __init__(self, env, simulation_app=None, robot_key_handler=None, enabled=True):
        self.env = env
        self.simulation_app = simulation_app
        self.robot_key_handler = robot_key_handler
        self.cfg = env.cfg.viewer_control
        self.video_cfg = env.cfg.video
        self.enabled = bool(enabled)
        self.paused = False
        self.stop_requested = False
        self.reset_requested = False
        self.enable_viewer_sync = bool(enabled)
        self.free_cam = False
        self.lookat_id = max(0, min(int(self.cfg.ref_env), int(env.num_envs) - 1))
        self.lookat_vec_local = torch.tensor(self.cfg.follow_vec_local, device=env.device, dtype=torch.float32)
        self.lookat_vec_local_cpu = [float(v) for v in self.cfg.follow_vec_local]
        self.lookat_follow_yaw = 0.0
        self._keyboard_sub = None
        self._last_repeat = {}
        if self.enabled:
            self._set_initial_camera()

    def _set_initial_camera(self):
        self.set_camera(self.cfg.pos, self.cfg.lookat)
        self._reset_follow_yaw_anchor(self.lookat_id)

    def install_keyboard(self):
        if not self.enabled:
            return self
        try:
            import carb
            import carb.input
            import omni.appwindow
        except Exception as exc:
            print(f"[viewer][warn] keyboard interface unavailable: {exc}")
            return self

        app_window = omni.appwindow.get_default_app_window()
        if app_window is None:
            print("[viewer][warn] no default app window; keyboard input unavailable.")
            return self
        keyboard = app_window.get_keyboard()
        input_iface = carb.input.acquire_input_interface()
        if keyboard is None or input_iface is None:
            print("[viewer][warn] keyboard/input interface unavailable.")
            return self

        def on_key_event(event, *args, **kwargs):
            if self._event_type_is_release(event.type):
                return True
            key = self._keyboard_input_to_key(event.input)
            if key is None:
                return True
            self.handle_key(key)
            return True

        self._keyboard_sub = input_iface.subscribe_to_keyboard_events(keyboard, on_key_event)
        print("[viewer] IsaacLab viewer keyboard subscribed.")
        return self

    def _event_type_is_release(self, event_type) -> bool:
        try:
            import carb

            keyboard_event_type = getattr(carb.input, "KeyboardEventType", None)
            release = getattr(keyboard_event_type, "KEY_RELEASE", None) if keyboard_event_type else None
            if release is not None:
                try:
                    return int(event_type) == int(release)
                except Exception:
                    return event_type == release
        except Exception:
            pass
        try:
            return int(event_type) == 0
        except Exception:
            return False

    def _keyboard_input_to_key(self, input_value) -> str | None:
        try:
            import carb

            keyboard_input = getattr(carb.input, "KeyboardInput", None)
            aliases = {
                "ESCAPE": "esc",
                "SPACE": "space",
                "V": "v",
                "F": "f",
                "KEY_0": "0",
                "KEY_1": "1",
                "KEY_2": "2",
                "KEY_3": "3",
                "KEY_4": "4",
                "KEY_5": "5",
                "KEY_6": "6",
                "KEY_7": "7",
                "KEY_8": "8",
                "KEY_9": "9",
                "LEFT_BRACKET": "[",
                "RIGHT_BRACKET": "]",
                "COMMA": ",",
                "PERIOD": ".",
                "LEFT": "left",
                "RIGHT": "right",
                "UP": "up",
                "DOWN": "down",
                "PAGE_UP": "pageup",
                "PAGE_DOWN": "pagedown",
            }
            if keyboard_input is not None:
                for attr, key in aliases.items():
                    enum_value = getattr(keyboard_input, attr, None)
                    if enum_value is None:
                        continue
                    try:
                        if int(input_value) == int(enum_value):
                            return key
                    except Exception:
                        if input_value == enum_value:
                            return key
        except Exception:
            pass

        text = str(input_value).strip().lower()
        text = text.replace("keyboardinput.", "").replace("key_", "")
        replacements = {
            "escape": "esc",
            "left_bracket": "[",
            "right_bracket": "]",
            "comma": ",",
            "period": ".",
            "page_up": "pageup",
            "page_down": "pagedown",
        }
        text = replacements.get(text, text)
        if text in {"esc", "space", "v", "f", "[", "]", ",", ".", "left", "right", "up", "down", "pageup", "pagedown"}:
            return text
        if len(text) == 1 and (text.isdigit() or text.isalpha()):
            return text
        return None

    def set_camera(self, position, lookat):
        if not self.enabled:
            return
        self.env.sim.set_camera_view(tuple(float(v) for v in position), tuple(float(v) for v in lookat))

    def _root_pos(self, env_id):
        return self.env.root_states[env_id, :3]

    def _root_state_cpu(self, env_id):
        return self.env.root_states[env_id, :7].detach().cpu().tolist()

    def _get_follow_yaw(self, env_id):
        root_state = self._root_state_cpu(env_id)
        w, x, y, z = root_state[3:7]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _wrap_to_pi(angle):
        return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi

    def _reset_follow_yaw_anchor(self, env_id):
        self.lookat_follow_yaw = self._get_follow_yaw(env_id)

    def _maybe_update_follow_yaw_anchor(self, env_id):
        current_yaw = self._get_follow_yaw(env_id)
        threshold = math.radians(float(self.cfg.follow_yaw_update_threshold_deg))
        if abs(self._wrap_to_pi(current_yaw - self.lookat_follow_yaw)) >= threshold:
            self.lookat_follow_yaw = current_yaw

    def _follow_vector_local_to_world(self, vec_local, yaw):
        cos_yaw = math.cos(float(yaw))
        sin_yaw = math.sin(float(yaw))
        return [
            cos_yaw * float(vec_local[0]) - sin_yaw * float(vec_local[1]),
            sin_yaw * float(vec_local[0]) + cos_yaw * float(vec_local[1]),
            float(vec_local[2]),
        ]

    def lookat(self, env_id=None, force=False):
        if env_id is not None:
            self.lookat_id = max(0, min(int(env_id), int(self.env.num_envs) - 1))
        self._maybe_update_follow_yaw_anchor(self.lookat_id)
        root_state = self._root_state_cpu(self.lookat_id)
        target = root_state[:3]
        follow_vec_world = self._follow_vector_local_to_world(self.lookat_vec_local_cpu, self.lookat_follow_yaw)
        position = [target[i] + follow_vec_world[i] for i in range(3)]
        self.set_camera(position, target)

    def _orbit_camera(self, delta_yaw=0.0, delta_pitch=0.0, delta_radius=0.0):
        radius = max(float(torch.norm(self.lookat_vec_local).item()), float(self.cfg.orbit_radius_min))
        yaw = math.atan2(self.lookat_vec_local[1].item(), self.lookat_vec_local[0].item())
        pitch = math.asin(max(-1.0, min(1.0, self.lookat_vec_local[2].item() / max(radius, 1.0e-6))))
        radius = max(float(self.cfg.orbit_radius_min), radius + float(delta_radius))
        pitch_limit = math.radians(float(self.cfg.orbit_pitch_limit_deg))
        pitch = max(-pitch_limit, min(pitch_limit, pitch + float(delta_pitch)))
        yaw += float(delta_yaw)
        cos_pitch = math.cos(pitch)
        self.lookat_vec_local = torch.tensor(
            [
                radius * cos_pitch * math.cos(yaw),
                radius * cos_pitch * math.sin(yaw),
                radius * math.sin(pitch),
            ],
            device=self.env.device,
            dtype=torch.float32,
        )
        self.lookat_vec_local_cpu = [float(v) for v in self.lookat_vec_local.detach().cpu().tolist()]
        self.lookat(force=True)

    def _repeat_allowed(self, key):
        if key not in {"[", "]", "left", "right", "up", "down", "pageup", "pagedown"}:
            return True
        now = time.monotonic()
        repeat_period = 1.0 / max(float(self.env.cfg.env.teleop_key_repeat_rate_hz), 1.0e-6)
        next_time = self._last_repeat.get(key, 0.0)
        if now < next_time:
            return False
        self._last_repeat[key] = now + repeat_period
        return True

    def handle_key(self, key):
        if not self.enabled:
            return
        if not self._repeat_allowed(key):
            return
        if key == "esc":
            self.stop_requested = True
            if self.simulation_app is not None:
                self.simulation_app.close()
            return
        if key == "v":
            self.enable_viewer_sync = not self.enable_viewer_sync
            print(f"[viewer] sync={self.enable_viewer_sync}")
            return
        if key == "f":
            self.free_cam = not self.free_cam
            print(f"[viewer] mode={'free' if self.free_cam else 'follow'}")
            if not self.free_cam:
                self._reset_follow_yaw_anchor(self.lookat_id)
                self.lookat(force=True)
            return
        if key == "space":
            self.paused = not self.paused
            print(f"[viewer] paused={self.paused}")
            return
        if key == "9":
            self.reset_requested = True
            print("[viewer] manual reset requested for all environments")
            return
        if not self.free_cam:
            if key.isdigit() and key != "9":
                env_id = int(key)
                if env_id < min(9, int(self.env.num_envs)):
                    self._reset_follow_yaw_anchor(env_id)
                    self.lookat(env_id, force=True)
                return
            if key == "[":
                self.lookat_id = (self.lookat_id - 1) % int(self.env.num_envs)
                self._reset_follow_yaw_anchor(self.lookat_id)
                self.lookat(force=True)
                return
            if key == "]":
                self.lookat_id = (self.lookat_id + 1) % int(self.env.num_envs)
                self._reset_follow_yaw_anchor(self.lookat_id)
                self.lookat(force=True)
                return
            yaw_step = math.radians(float(self.cfg.orbit_yaw_step_deg))
            pitch_step = math.radians(float(self.cfg.orbit_pitch_step_deg))
            radius_step = float(self.cfg.orbit_radius_step)
            if key == "left":
                self._orbit_camera(delta_yaw=-yaw_step)
                return
            if key == "right":
                self._orbit_camera(delta_yaw=yaw_step)
                return
            if key == "up":
                self._orbit_camera(delta_pitch=pitch_step)
                return
            if key == "down":
                self._orbit_camera(delta_pitch=-pitch_step)
                return
            if key == "pageup":
                self._orbit_camera(delta_radius=-radius_step)
                return
            if key == "pagedown":
                self._orbit_camera(delta_radius=radius_step)
                return
        if self.robot_key_handler is not None:
            self.robot_key_handler(key)

    def consume_reset_request(self):
        requested = self.reset_requested
        self.reset_requested = False
        return requested

    def tick(self):
        if not self.enabled or not self.enable_viewer_sync:
            return
        if not self.free_cam:
            self.lookat()
