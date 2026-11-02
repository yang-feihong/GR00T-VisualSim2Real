"""Restore batched canonical pick states for camera-only official collection."""

import json
import shutil

import torch

from gr00t.rl.data_collection.auto_open_standard import (
    camera_streams_for_mode,
    transform_collection_images,
)
from gr00t.rl.data_collection.auto_open_writer import StreamingVideoEncoder, _json_dump, file_sha256
from gr00t.rl.pick.dataset import episode_dataset_root
from gr00t.rl.pick.official import validate_canonical_source


def replay_frame_batch(
    records, lengths, frame, episodes, distractor_template, scene_translation=None
):
    """Discard source-rank padding and use this rank's inert padding states."""
    pre = {}
    for key in records[0]["pre"]:
        if key == "distractor_states":
            states = distractor_template.clone()
            for row, (record, length, episode) in enumerate(zip(records, lengths, episodes)):
                count = len(episode.get("distractors", []))
                source = record["pre"][key][min(frame, length - 1)]
                if count > len(source) or count > states.shape[1]:
                    raise ValueError("Canonical distractor state count differs from its descriptor")
                states[row, :count] = source[:count]
                if scene_translation is not None:
                    states[row, :count, :3] += scene_translation[row]
            pre[key] = states[None]
        else:
            pre[key] = torch.stack(
                [
                    record["pre"][key][min(frame, length - 1)]
                    for record, length in zip(records, lengths)
                ]
            )[None]
            if scene_translation is not None and key in ("root_states", "object_state"):
                pre[key][0, :, :3] += scene_translation
    return pre


def restore_frame(env, pre, index):
    """Restore all supplied rows together, without a simulator or policy step."""

    def at(key):
        return pre[key][index] if pre["root_states"].ndim == 3 else pre[key][index : index + 1]

    env.robot.write_root_state_to_sim(at("root_states"))
    env.robot.write_joint_state_to_sim(at("dof_pos"), at("dof_vel"))
    env.pick_object.write_root_state_to_sim(at("object_state"))
    for slot, obj in enumerate(env.pick_distractors):
        obj.write_root_state_to_sim(at("distractor_states")[:, slot])
    env.sim.forward()
    env.robot.update(float(env.dt))
    env.pick_object.update(float(env.dt))
    for obj in env.pick_distractors:
        obj.update(float(env.dt))


def state_restore_error(env, pre, index):
    def at(key):
        return pre[key][index] if pre["root_states"].ndim == 3 else pre[key][index : index + 1]

    def rigid_error(actual, expected):
        expected = expected.clone()
        flip = (actual[:, 3:7] * expected[:, 3:7]).sum(-1) < 0
        expected[:, 3:7] = torch.where(flip[:, None], -expected[:, 3:7], expected[:, 3:7])
        return maximum_error(actual, expected)

    def maximum_error(actual, expected):
        return (actual - expected).abs().nan_to_num(nan=float("inf"), posinf=float("inf")).max()

    error = rigid_error(env.robot.data.root_state_w, at("root_states"))
    for actual, expected in (
        (env.pick_object.data.root_state_w, at("object_state")),
        (env.robot.data.joint_pos, at("dof_pos")),
        (env.robot.data.joint_vel, at("dof_vel")),
    ):
        difference = (
            rigid_error(actual, expected)
            if actual.shape[-1] == 13
            else maximum_error(actual, expected)
        )
        error = torch.maximum(error, difference)
    for slot, obj in enumerate(env.pick_distractors):
        error = torch.maximum(
            error, rigid_error(obj.data.root_state_w, at("distractor_states")[:, slot])
        )
    return float(error)


def render_shared_rank(args, runner, episodes):
    env, manager = runner.env, runner.rgb_camera_manager
    sources, fingerprints, records, lengths, source_origins = [], [], [], [], []
    for episode in episodes:
        seed = episode["seed"]
        matches = list(args.replay_source_dataset_dir.glob(f"logs/rank*-control/seed_{seed}"))
        if len(matches) != 1:
            raise ValueError(f"Canonical scene is not unique for seed {seed}")
        source = matches[0]
        sources.append(source)
        descriptor = json.loads((source / "episode.json").read_text())
        source_origins.append(descriptor.get("batch_scene_origin_m", [0.0, 0.0, 0.0]))
        fingerprints.append(validate_canonical_source(source))
        record = torch.load(
            source / "control_record.pt", map_location=env.device, weights_only=False
        )
        records.append(record)
        lengths.append(len(record["observation_state"]))
    layout = camera_streams_for_mode("dual")
    encoders = [
        {
            stream["key"]: StreamingVideoEncoder(
                args.output_dir / f"seed_{episode['seed']}/.video_staging/{name}.mp4",
                stream["shape"],
            )
            for name, stream in layout.items()
        }
        for episode in episodes
    ]
    maximum_error = 0.0
    distractor_template = env.get_control_state()["distractor_states"].clone()
    translation = env.scene.env_origins - env.scene.env_origins.new_tensor(source_origins)
    try:
        for frame in range(max(lengths)):
            pre = replay_frame_batch(
                records, lengths, frame, episodes, distractor_template, translation
            )
            restore_frame(env, pre, 0)
            if frame % 50 == 0 or frame == max(lengths) - 1:
                error = state_restore_error(env, pre, 0)
                maximum_error = max(maximum_error, error)
                if error > 1e-5:
                    raise RuntimeError(
                        f"Canonical replay state mismatch: frame={frame} error={error}"
                    )
                (args.output_dir / "progress.txt").write_text(
                    f"stage=CAMERA_REPLAY frame={frame}/{max(lengths)}\n"
                )
            active_rows = [row for row, length in enumerate(lengths) if frame < length]
            frames = manager.render_images_batch(active_rows, rig_names=list(layout))
            for row in active_rows:
                rendered = frames[row]
                images = transform_collection_images(
                    {name: rendered[name][0] for name in layout}, "dual"
                )
                for key, image in images.items():
                    encoders[row][key].append(image)
        for group in encoders:
            for encoder in group.values():
                encoder.close()
        for row, (episode, source) in enumerate(zip(episodes, sources)):
            seed = episode["seed"]
            if validate_canonical_source(source) != fingerprints[row]:
                raise RuntimeError("Canonical source changed during camera replay")
            source_dataset = episode_dataset_root(source)
            shard = args.standard_dataset_dir / f"episode-seed{seed}"
            shutil.copytree(source_dataset, shard)
            info = json.loads((shard / "meta/info.json").read_text())
            for stream in layout.values():
                key = stream["key"]
                destination = shard / "videos" / key / "chunk-000/episode_000000.mp4"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(encoders[row][key].path, destination)
                info["features"][key] = {"dtype": "video", "shape": list(stream["shape"])}
            _json_dump(shard / "meta/info.json", info)
            provenance = json.loads((shard / "meta/simulation_provenance.json").read_text())
            provenance.update(
                mode="official",
                camera_mode="dual",
                images_enabled=True,
                physics_steps_during_replay=0,
                canonical_sha256=fingerprints[row],
            )
            _json_dump(shard / "meta/simulation_provenance.json", provenance)
            _json_dump(
                shard / "meta/camera_calibration.json", args.camera_mount_randomizations[seed]
            )
            for relative in ("data", "raw_audit", "raw_contacts"):
                file = f"{relative}/chunk-000/episode_000000.parquet"
                if file_sha256(shard / file) != file_sha256(source_dataset / file):
                    raise RuntimeError(f"Camera replay changed canonical table: {file}")
            _json_dump(
                args.output_dir / f"seed_{seed}/summary.json",
                {
                    "seed": seed,
                    "success": True,
                    "mode": "camera_replay",
                    "physics_steps": 0,
                    "frames": lengths[row],
                    "maximum_state_restore_error": maximum_error,
                },
            )
        _json_dump(
            args.output_dir / "summary.json",
            {
                "episode_seeds": [episode["seed"] for episode in episodes],
                "physics_steps": 0,
                "mode": "camera_replay",
                "maximum_state_restore_error": maximum_error,
            },
        )
    finally:
        for group in encoders:
            for encoder in group.values():
                encoder.close()
