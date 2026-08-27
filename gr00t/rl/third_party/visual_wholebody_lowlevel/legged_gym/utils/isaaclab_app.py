from __future__ import annotations

import os
import sys
import argparse

import torch

_simulation_app = None

_VIEWPORT_DISPLAY_FPS = 1 << 0
_VIEWPORT_DISPLAY_RESOLUTION = 1 << 3
_VIEWPORT_DISPLAY_MESH = 1 << 10
_VIEWPORT_DISPLAY_DEV_MEM = 1 << 13
_VIEWPORT_DISPLAY_HOST_MEM = 1 << 14
_VIEWPORT_PERFORMANCE_HUD_OPTIONS = (
    _VIEWPORT_DISPLAY_FPS
    | _VIEWPORT_DISPLAY_RESOLUTION
    | _VIEWPORT_DISPLAY_MESH
    | _VIEWPORT_DISPLAY_DEV_MEM
    | _VIEWPORT_DISPLAY_HOST_MEM
)
_RENDERER_MULTI_GPU_KIT_ARGS = (
    "--/renderer/multiGpu/enabled=true",
    "--/renderer/multiGPU/enabled=true",
    "--/renderer/multiGpu/autoEnable=true",
    "--/renderer/multiGPU/autoEnable=true",
)
_renderer_feature_settings = None


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {value!r}")


def add_app_launcher_args(parser):
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    if "--no-headless" not in parser._option_string_actions:
        parser.add_argument("--no-headless", dest="headless", action="store_false", default=argparse.SUPPRESS)
    renderer_group = parser.add_argument_group("RTX renderer features")
    renderer_group.add_argument("--rtx_reflections", type=_parse_bool, default=False)
    renderer_group.add_argument("--rtx_translucency", type=_parse_bool, default=False)
    renderer_group.add_argument("--rtx_subsurface_scattering", type=_parse_bool, default=False)
    renderer_group.add_argument("--rtx_caustics", type=_parse_bool, default=False)
    renderer_group.add_argument("--rtx_indirect_diffuse", type=_parse_bool, default=False)
    renderer_group.add_argument("--rtx_ambient_occlusion", type=_parse_bool, default=False)
    renderer_group.add_argument("--rtx_direct_lighting_spp", type=int, default=1)
    renderer_group.add_argument("--rtx_reflections_spp", type=int, default=1)


def load_rgb_camera_specs(config_path):
    if not config_path:
        return None
    from rgb_camera_debug import load_rgb_camera_config, rgb_camera_specs_from_config

    return rgb_camera_specs_from_config(load_rgb_camera_config(config_path))


def _enable_viewport_performance_hud(args):
    if bool(args.headless):
        return
    args.display_options = _VIEWPORT_PERFORMANCE_HUD_OPTIONS


def _enable_renderer_multi_gpu(args):
    visible_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    renderer_gpu_count = max(int(visible_gpu_count), 1)
    device = str(args.device)
    active_gpu = int(device.split(":")[-1]) if device.startswith("cuda:") else 0
    args.multi_gpu = True
    for setting_arg in _RENDERER_MULTI_GPU_KIT_ARGS:
        if setting_arg not in sys.argv:
            sys.argv.append(setting_arg)
    for setting_arg in (
        f"--/renderer/multiGpu/maxGpuCount={renderer_gpu_count}",
        f"--/renderer/multiGPU/maxGpuCount={renderer_gpu_count}",
        f"--/renderer/activeGpu={active_gpu}",
    ):
        if setting_arg not in sys.argv:
            sys.argv.append(setting_arg)


def _sync_headless_env(args):
    os.environ["HEADLESS"] = "1" if bool(args.headless) else "0"


def _show_viewport_performance_hud():
    try:
        import carb.settings

        settings = carb.settings.get_settings()
        settings.set("/persistent/app/viewport/displayOptions", _VIEWPORT_PERFORMANCE_HUD_OPTIONS)
        settings.set("/app/viewport/displayOptions", _VIEWPORT_PERFORMANCE_HUD_OPTIONS)
    except Exception as exc:
        print(f"[viewer][warn] failed to set viewport display options: {exc}")


def _disable_viewport_updates_in_headless(args):
    if not bool(args.headless):
        return

    try:
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport is not None:
            viewport.updates_enabled = False
    except Exception as exc:
        print(f"[viewer][warn] failed to disable viewport updates in headless mode: {exc}")


def _apply_renderer_multi_gpu_settings():
    try:
        import carb.settings

        settings = carb.settings.get_settings()
        visible_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        renderer_gpu_count = max(int(visible_gpu_count), 1)
        for setting_path in (
            "/renderer/multiGpu/enabled",
            "/renderer/multiGpu/autoEnable",
            "/renderer/multiGPU/enabled",
            "/renderer/multiGPU/autoEnable",
        ):
            settings.set(setting_path, True)
        for setting_path in (
            "/renderer/multiGpu/maxGpuCount",
            "/renderer/multiGPU/maxGpuCount",
        ):
            settings.set(setting_path, renderer_gpu_count)
        print(f"[renderer] requested multi-GPU rendering with maxGpuCount={renderer_gpu_count}")
    except Exception as exc:
        print(f"[renderer][warn] failed to enable renderer multi-GPU: {exc}")


def _capture_renderer_feature_settings(args):
    global _renderer_feature_settings
    direct_spp = int(args.rtx_direct_lighting_spp)
    reflections_spp = int(args.rtx_reflections_spp)
    if direct_spp < 1 or reflections_spp < 1:
        raise ValueError("RTX samples per pixel must be at least 1")
    _renderer_feature_settings = {
        "/rtx/reflections/enabled": bool(args.rtx_reflections),
        "/rtx/translucency/enabled": bool(args.rtx_translucency),
        "/rtx/raytracing/subsurface/enabled": bool(args.rtx_subsurface_scattering),
        "/rtx/caustics/enabled": bool(args.rtx_caustics),
        "/rtx/indirectDiffuse/enabled": bool(args.rtx_indirect_diffuse),
        "/rtx/ambientOcclusion/enabled": bool(args.rtx_ambient_occlusion),
        "/rtx/directLighting/sampledLighting/samplesPerPixel": direct_spp,
        "/rtx/reflections/sampledLighting/samplesPerPixel": reflections_spp,
    }


def apply_renderer_feature_settings():
    """Reapply requested RTX features after SimulationContext loads its preset."""
    if _renderer_feature_settings is None:
        return
    try:
        import carb.settings

        settings = carb.settings.get_settings()
        for setting_path, value in _renderer_feature_settings.items():
            settings.set(setting_path, value)
        labels = {
            "/rtx/reflections/enabled": "reflections",
            "/rtx/translucency/enabled": "translucency",
            "/rtx/raytracing/subsurface/enabled": "subsurface_scattering",
            "/rtx/caustics/enabled": "caustics",
            "/rtx/indirectDiffuse/enabled": "indirect_diffuse",
            "/rtx/ambientOcclusion/enabled": "ambient_occlusion",
            "/rtx/directLighting/sampledLighting/samplesPerPixel": "direct_lighting_spp",
            "/rtx/reflections/sampledLighting/samplesPerPixel": "reflections_spp",
        }
        summary = ", ".join(f"{labels[path]}={value}" for path, value in _renderer_feature_settings.items())
        print(f"[renderer] applied RTX camera settings: {summary}")
    except Exception as exc:
        print(f"[renderer][warn] failed to apply RTX camera settings: {exc}")


def _rgb_camera_config_has_enabled_cameras(config_path: str) -> bool:
    try:
        parsed = load_rgb_camera_specs(config_path)
    except Exception:
        return bool(config_path)
    return bool(parsed and parsed["enabled"] and len(parsed["cameras"]) > 0)


def launch_app(args):
    global _simulation_app
    if _simulation_app is None:
        from isaaclab.app import AppLauncher

        needs_cameras = (
            _rgb_camera_config_has_enabled_cameras(args.rgb_camera_config)
            or bool(args.record_video)
            or bool(args.xr_publish_rgb_camera)
        )
        if needs_cameras and hasattr(args, "enable_cameras"):
            args.enable_cameras = True
        _capture_renderer_feature_settings(args)
        _sync_headless_env(args)
        _enable_viewport_performance_hud(args)
        _enable_renderer_multi_gpu(args)
        _simulation_app = AppLauncher(args).app
        _apply_renderer_multi_gpu_settings()
        _disable_viewport_updates_in_headless(args)
        if not bool(args.headless):
            _show_viewport_performance_hud()
    return _simulation_app
