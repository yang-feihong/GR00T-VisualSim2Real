"""The shared official workflow must restore pick states without physics."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gr00t.rl.pick.official import validate_canonical_source
from gr00t.rl.pick.prepare_batch import local_episode_descriptor
from gr00t.rl.pick.replay import (
    render_shared_rank,
    replay_frame_batch,
    restore_frame,
    state_restore_error,
)
from gr00t.rl.pick.tensor_planning import episode_in_world


@pytest.mark.parametrize("count", [1, 3])
def test_replay_restores_all_rows_without_physics(count):
    calls = []

    class Body:
        def __init__(self):
            self.data = SimpleNamespace()

        def write_root_state_to_sim(self, value):
            self.data.root_state_w = value.clone()

        def write_joint_state_to_sim(self, q, qd):
            self.data.joint_pos, self.data.joint_vel = q.clone(), qd.clone()

        def update(self, dt):
            calls.append(("update", dt))

    env = SimpleNamespace(
        robot=Body(),
        pick_object=Body(),
        pick_distractors=[Body()],
        dt=0.02,
        sim=SimpleNamespace(
            forward=lambda: calls.append("forward"), step=lambda: pytest.fail("No replay physics")
        ),
    )
    pre = {
        "root_states": torch.zeros(1, count, 13),
        "dof_pos": torch.zeros(1, count, 19),
        "dof_vel": torch.ones(1, count, 19),
        "object_state": torch.zeros(1, count, 13),
        "distractor_states": torch.zeros(1, count, 1, 13),
    }
    for name in ("root_states", "object_state", "distractor_states"):
        pre[name][..., 3] = 1
    pre["root_states"][0, :, 0] = torch.arange(count)
    if count == 1:
        pre = {key: value[:, 0] for key, value in pre.items()}
    restore_frame(env, pre, 0)
    assert env.robot.data.root_state_w.shape == (count, 13)
    assert state_restore_error(env, pre, 0) == 0
    env.robot.data.root_state_w[:, 3:7] *= -1
    assert state_restore_error(env, pre, 0) == 0
    env.pick_distractors[0].data.root_state_w[:, 0] += 0.02
    assert state_restore_error(env, pre, 0) == pytest.approx(0.02, abs=1e-6)
    for value in (
        env.robot.data.root_state_w,
        env.robot.data.joint_pos,
        env.robot.data.joint_vel,
        env.pick_object.data.root_state_w,
        env.pick_distractors[0].data.root_state_w,
    ):
        saved = value[0, 0].clone()
        value[0, 0] = float("nan")
        assert state_restore_error(env, pre, 0) == float("inf")
        value[0, 0] = saved
    assert calls.count("forward") == 1


@pytest.mark.parametrize("mode,accepted", [("debug", True), ("canonical", False)])
def test_official_rejects_unqualified_sources_before_rendering(tmp_path, mode, accepted):
    (tmp_path / "summary.json").write_text(json.dumps({"mode": mode, "success": accepted}))
    meta = tmp_path / "dataset/meta"
    meta.mkdir(parents=True)
    (meta / "acceptance_report.json").write_text(
        json.dumps({"accepted_episodes": int(accepted), "quarantined_episodes": int(not accepted)})
    )
    with pytest.raises(ValueError, match="strictly accepted"):
        validate_canonical_source(tmp_path)


def test_canonical_world_descriptor_is_localized_exactly_once():
    original = {
        "object_position": [1.0, 2.0, 0.5],
        "support_z": 0.4,
        "base_xy": [2.0, 3.0],
        "station_xy": [1.5, 3.0],
        "distractors": [{"object_position": [3.0, 4.0, 0.6], "support_z": 0.5}],
    }
    world = episode_in_world(original, torch.tensor([32.0, 64.0, 1.0]))
    local = local_episode_descriptor(world)
    assert local["object_position"] == pytest.approx(original["object_position"])
    assert local["support_z"] == pytest.approx(original["support_z"])
    assert local["base_xy"] == original["base_xy"]
    assert local["station_xy"] == original["station_xy"]
    assert local["distractors"][0]["object_position"] == pytest.approx(
        original["distractors"][0]["object_position"]
    )
    assert local_episode_descriptor(local) == local
    assert world["object_position"] == [33.0, 66.0, 1.5]
    assert original["object_position"] == [1.0, 2.0, 0.5]


def test_camera_replay_repartitions_distractors_without_importing_source_padding():
    records = [
        {
            "pre": {
                "root_states": torch.zeros(2, 13),
                "distractor_states": torch.full((2, 1, 13), 9.0),
            }
        },
        {
            "pre": {
                "root_states": torch.ones(1, 13),
                "distractor_states": torch.full((1, 3, 13), 5.0),
            }
        },
    ]
    template = torch.full((2, 2, 13), -100.0)
    pre = replay_frame_batch(records, [2, 1], 1, [{}, {"distractors": [{}, {}]}], template)
    assert pre["root_states"].shape == (1, 2, 13)
    torch.testing.assert_close(pre["root_states"][0, 1], records[1]["pre"]["root_states"][0])
    assert (pre["distractor_states"][0, 0] == -100).all()
    assert (pre["distractor_states"][0, 1] == 5).all()
    assert (template == -100).all()
    assert (records[0]["pre"]["distractor_states"] == 9).all()
    with pytest.raises(ValueError, match="distractor state count"):
        replay_frame_batch(records, [2, 1], 1, [{"distractors": [{}, {}]}, {}], template)


def test_replay_relocates_physical_states_when_rank_scene_origins_change():
    record = {
        "pre": {
            key: torch.arange(26, dtype=torch.float32).reshape(2, 13)
            for key in ("root_states", "object_state", "dof_pos")
        }
    }
    record["pre"]["distractor_states"] = torch.ones(2, 1, 13)
    originals = {key: value.clone() for key, value in record["pre"].items()}
    template = torch.full((2, 1, 13), -100.0)
    translation = torch.tensor([[32.0, -64.0, 1.0], [-32.0, 0.0, 0.0]])
    pre = replay_frame_batch(
        [record, record], [2, 2], 0, [{"distractors": [{}]}, {}], template, translation
    )
    for key in ("root_states", "object_state"):
        torch.testing.assert_close(pre[key][0, :, :3], originals[key][0, :3] + translation)
        torch.testing.assert_close(pre[key][0, :, 3:], originals[key][0, 3:].expand(2, -1))
    torch.testing.assert_close(pre["dof_pos"][0], originals["dof_pos"][:1].expand(2, -1))
    torch.testing.assert_close(pre["distractor_states"][0, 0, 0, :3], 1 + translation[0])
    assert (pre["distractor_states"][0, 1] == -100).all()
    for key, value in originals.items():
        torch.testing.assert_close(record["pre"][key], value)


def test_official_rank_replays_unequal_lengths_and_repartitioned_scenes_without_physics(
    tmp_path, monkeypatch
):
    from gr00t.rl.pick import replay

    class Body:
        def __init__(self):
            self.data = SimpleNamespace()

        def write_root_state_to_sim(self, value):
            self.data.root_state_w = value.clone()

        def write_joint_state_to_sim(self, q, qd):
            self.data.joint_pos, self.data.joint_vel = q.clone(), qd.clone()

        def update(self, dt):
            pass

    sources = tmp_path / "canonical"
    episodes = [{"seed": 100, "distractors": [{}]}, {"seed": 101}]
    origins = torch.tensor([[0.0, 32, 0], [64.0, 0, 0]])
    old_origins = torch.tensor([[32.0, 0, 0], [0.0, 64, 0]])
    for row, (episode, length) in enumerate(zip(episodes, [3, 1])):
        source = sources / f"logs/rank{row}-control/seed_{episode['seed']}"
        dataset = source / "dataset"
        (dataset / "meta").mkdir(parents=True)
        (source / "summary.json").write_text(json.dumps({"mode": "canonical", "success": True}))
        (source / "episode.json").write_text(
            json.dumps({**episode, "batch_scene_origin_m": old_origins[row].tolist()})
        )
        for name, value in (
            ("acceptance_report.json", {"accepted_episodes": 1, "quarantined_episodes": 0}),
            ("info.json", {"features": {}}),
            ("simulation_provenance.json", {"mode": "canonical"}),
        ):
            (dataset / "meta" / name).write_text(json.dumps(value))
        for name in ("data", "raw_audit", "raw_contacts"):
            table = dataset / name / "chunk-000/episode_000000.parquet"
            table.parent.mkdir(parents=True)
            table.write_bytes(f"immutable-{episode['seed']}-{name}".encode())
        root = torch.zeros(length, 13)
        root[:, :3] = old_origins[row]
        root[:, 3] = 1
        obj = root.clone()
        obj[:, 2] = torch.arange(length) + 0.5
        pre = {
            "root_states": root,
            "object_state": obj,
            "dof_pos": torch.zeros(length, 19),
            "dof_vel": torch.zeros(length, 19),
            "distractor_states": root[:, None].expand(-1, row + 1, -1).clone(),
        }
        torch.save(
            {"pre": pre, "observation_state": torch.zeros(length, 58)},
            source / "control_record.pt",
        )

    encoded, renders, forwards = {}, [], []

    class Encoder:
        def __init__(self, path, shape):
            self.path = path
            path.parent.mkdir(parents=True, exist_ok=True)
            self.frames = encoded.setdefault(str(path), [])
            self.closed = False

        def append(self, image):
            self.frames.append(image.copy())

        def close(self):
            if not self.closed:
                self.path.write_bytes(b"offline-video-placeholder")
                self.closed = True

    monkeypatch.setattr(replay, "StreamingVideoEncoder", Encoder)
    monkeypatch.setattr(
        replay,
        "transform_collection_images",
        lambda images, mode: {layout[name]["key"]: image for name, image in images.items()},
    )
    env = SimpleNamespace(
        device="cpu",
        dt=0.02,
        robot=Body(),
        pick_object=Body(),
        pick_distractors=[Body()],
        scene=SimpleNamespace(env_origins=origins),
        sim=SimpleNamespace(
            forward=lambda: forwards.append(True), step=lambda: pytest.fail("Replay ran physics")
        ),
        get_control_state=lambda: {"distractor_states": torch.full((2, 1, 13), -100.0)},
    )
    layout = replay.camera_streams_for_mode("dual")

    def render(rows, rig_names):
        renders.append(rows)
        torch.testing.assert_close(env.robot.data.root_state_w[:, :3], origins)
        torch.testing.assert_close(env.pick_object.data.root_state_w[:, :2], origins[:, :2])
        assert (env.pick_distractors[0].data.root_state_w[1] == -100).all()
        return {
            row: {name: [np.full((2, 2, 3), row, dtype="uint8")] for name in layout} for row in rows
        }

    args = SimpleNamespace(
        replay_source_dataset_dir=sources,
        output_dir=tmp_path / "render",
        standard_dataset_dir=tmp_path / "shards",
        camera_mount_randomizations={100: {"seed": 100}, 101: {"seed": 101}},
    )
    render_shared_rank(
        args,
        SimpleNamespace(env=env, rgb_camera_manager=SimpleNamespace(render_images_batch=render)),
        episodes,
    )
    assert renders == [[0, 1], [0], [0]]
    assert len(forwards) == 3
    assert sorted(map(len, encoded.values())) == [1, 1, 3, 3]
    for episode, length in zip(episodes, [3, 1]):
        summary = json.loads((args.output_dir / f"seed_{episode['seed']}/summary.json").read_text())
        assert summary["success"] and summary["frames"] == length
        assert summary["maximum_state_restore_error"] == 0
