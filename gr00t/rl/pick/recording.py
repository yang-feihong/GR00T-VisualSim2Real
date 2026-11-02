"""Preserve simulation time when scheduled camera updates have timing jitter."""

import argparse
import json
from pathlib import Path

import numpy as np


def read_control_camera_frames(manager, sim_time):
    """Capture both training cameras at every pre-control state; diagnostics keep their cadence."""
    primary = ("base_0_rgb", "left_wrist_0_rgb")
    due = manager.due_rigs_at_sim_time(sim_time)
    names = list(dict.fromkeys([*primary, *[rig["name"] for rig in due]]))
    rendered = manager.render_images(env_id=0, rig_names=names, update=True)
    frames = {}
    for name in names:
        values = rendered.get(name, [])
        if len(values) != 1:
            raise RuntimeError(f"Missing pre-control image at {sim_time}: {name}")
        frames[name] = values[0]
    for rig in manager.rigs:
        if rig["name"] in frames:
            manager.mark_published_at_sim_time(rig, sim_time)
    return frames


def read_due_camera_frames(manager, sim_time):
    """Render-only camera sidecars must be refreshed explicitly before reading."""
    if manager is None or not manager.enabled:
        return {}
    due = manager.due_rigs_at_sim_time(sim_time)
    if not due:
        return {}
    images = manager.render_images(env_id=0, rig_names=[rig["name"] for rig in due], update=True)
    frames = {}
    for rig in due:
        values = images.get(rig["name"], [])
        for index, frame in enumerate(values):
            name = rig["name"] if len(values) == 1 else f"{rig['name']}_{index}"
            frames[name] = frame
        if values:
            manager.mark_published_at_sim_time(rig, sim_time)
    return frames


def uniform_camera_streams(frames, timestamps):
    output_frames, output_times = {}, {}
    for name, source in frames.items():
        times = np.asarray(timestamps[name], dtype=np.float64)
        if len(source) != len(times) or not len(times):
            raise ValueError(f"Mismatched or empty camera stream: {name}")
        if len(times) == 1:
            output_frames[name], output_times[name] = source, times.tolist()
            continue
        intervals = np.diff(times)
        if (intervals <= 0).any():
            raise ValueError(f"Non-increasing camera timestamps: {name}")
        period = float(np.median(intervals))
        timeline = times[0] + np.arange(int(np.ceil((times[-1] - times[0]) / period)) + 1) * period
        right = np.searchsorted(times, timeline).clip(0, len(times) - 1)
        left = (right - 1).clip(0)
        indices = np.where(timeline - times[left] <= times[right] - timeline, left, right)
        output_frames[name] = [source[index] for index in indices]
        output_times[name] = timeline.tolist()
    return output_frames, output_times


def main():
    """Repair an existing debug recording without restarting the simulator."""
    import imageio.v2 as imageio

    from gr00t.rl.auto_open.io import save_rgb_mosaic_video, save_rgb_video

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    timestamps = json.loads((args.run / "camera_timestamps.json").read_text())
    frames = {}
    for name in timestamps:
        with imageio.get_reader(str(args.run / f"{name}.mp4")) as reader:
            frames[name] = [frame for frame in reader]
    frames, times = uniform_camera_streams(frames, timestamps)
    for name in frames:
        save_rgb_video(frames[name], times[name], args.run / f"{name}_retimed.mp4")
    save_rgb_mosaic_video(frames, times, args.run / "six_camera_mosaic_retimed.mp4")
    (args.run / "video_timestamps.json").write_text(json.dumps(times) + "\n")
    print(f"[pick:recording] {args.run / 'six_camera_mosaic_retimed.mp4'}", flush=True)


if __name__ == "__main__":
    main()
