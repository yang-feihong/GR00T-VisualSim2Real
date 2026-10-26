import copy
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gr00t.rl.data_collection.fisheye_border import FisheyeBorderAugmenter, STYLES


SPEC = {
    "projection_type": "opencvFisheye",
    "width": 80,
    "height": 60,
    "fx": 33.0,
    "fy": 34.0,
    "cx": 41.0,
    "cy": 31.0,
    "valid_circle_mask": {"center": [40.0, 30.0], "radius": 32.0, "feather": 2.0},
}


@pytest.mark.parametrize("feather", [0.0, 2.0])
def test_scene_feather_boundary_and_calibration_unchanged(feather):
    spec = copy.deepcopy(SPEC)
    spec["valid_circle_mask"]["feather"] = feather
    before = copy.deepcopy(spec)
    frame = np.random.default_rng(3).integers(0, 256, (60, 80, 3), dtype=np.uint8)
    original = frame.copy()
    yy, xx = np.mgrid[:60, :80]
    radius = np.hypot(xx - 40, yy - 30)
    valid = radius < 32 if feather else radius <= 32
    for seed in range(12):
        output = FisheyeBorderAugmenter(seed, "wrist", spec).apply(frame)
        np.testing.assert_array_equal(output[valid], frame[valid])
        assert output[~valid].max() <= 80
    np.testing.assert_array_equal(frame, original)
    assert spec == before


def test_independent_rng_temporal_seed_camera_and_batch_order_invariance():
    frame = np.full((60, 80, 3), 100, np.uint8)
    keys = [(seed, name) for seed in range(8) for name in ("wrist", "trunk_front")]
    forward = {key: FisheyeBorderAugmenter(*key, SPEC).apply(frame) for key in keys}
    np.random.seed(910)
    state = np.random.get_state()
    reverse = {key: FisheyeBorderAugmenter(*key, SPEC).apply(frame) for key in reversed(keys)}
    after = np.random.get_state()
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]
    for key in keys:
        np.testing.assert_array_equal(forward[key], reverse[key])
    assert len({value.tobytes() for value in forward.values()}) > 8
    augmenter = FisheyeBorderAugmenter(5, "wrist", SPEC)
    for _ in range(3):
        np.testing.assert_array_equal(augmenter.apply(frame), augmenter.apply(frame))
    other_scene = np.zeros_like(frame)
    mask = augmenter._invalid
    np.testing.assert_array_equal(augmenter.apply(frame)[mask], augmenter.apply(other_scene)[mask])


def test_texture_diversity_and_serializable_actual_parameters():
    frame = np.zeros((60, 80, 3), np.uint8)
    seen = set()
    for seed in range(100):
        augmenter = FisheyeBorderAugmenter(seed, "wrist", SPEC)
        metadata = json.loads(json.dumps(augmenter.metadata))
        style = metadata["style"]
        seen.add(style)
        border = augmenter.apply(frame)[augmenter._invalid]
        if style == "black":
            assert not border.any()
        else:
            assert border.std() > 1.0
            assert border.max() > 0
        repeated = FisheyeBorderAugmenter(seed, "wrist", SPEC)
        assert repeated.metadata == augmenter.metadata
        assert metadata["parameters"]["base_luminance"] <= 42
        metadata["parameters"]["base_luminance"] = -1
        assert augmenter.metadata["parameters"]["base_luminance"] >= 0
    assert seen == set(STYLES)


@pytest.mark.parametrize(
    "name,spec",
    [
        ("door_panel_diagnostic", SPEC),
        ("robot_top_diagnostic", SPEC),
        ("wrist", {**SPEC, "projection_type": "pinhole"}),
        ("wrist", {**SPEC, "valid_circle_mask": None}),
        ("wrist", {**SPEC, "valid_circle_mask": {"enabled": False}}),
    ],
)
def test_diagnostics_and_nonfisheye_are_untouched(name, spec):
    frame = np.ones((60, 80, 3), np.uint8)
    augmenter = FisheyeBorderAugmenter(5, name, spec)
    assert not augmenter.metadata["enabled"]
    assert augmenter.apply(frame) is frame


def test_effective_offset_mask_not_principal_point_or_black_detection():
    spec = copy.deepcopy(SPEC)
    spec["valid_circle_mask"] = {"center": [20, 25], "radius": 19, "feather": 3}
    frame = np.full((60, 80, 3), 150, np.uint8)
    augmenter = FisheyeBorderAugmenter(5, "wrist", spec)
    output = augmenter.apply(frame)
    assert np.all(output[25, 20] == 150)
    assert np.all(output[30, 40] <= 80)
    assert augmenter.metadata["valid_circle_mask"] == spec["valid_circle_mask"]
    with pytest.raises(ValueError, match="source-oriented"):
        augmenter.apply(frame.transpose(1, 0, 2))


def test_collector_dual_and_six_share_augmentation_and_preserve_diagnostics():
    from gr00t.rl.data_collection.auto_open_collector import AutoOpenStandardEpisodeCollector
    from gr00t.rl.data_collection.auto_open_standard import (
        camera_streams_for_mode,
        transform_collection_images,
    )

    spec = {
        **SPEC,
        "width": 640,
        "height": 480,
        "valid_circle_mask": {"center": [320, 240], "radius": 262, "feather": 2},
    }
    image = np.full((480, 640, 3), 120, np.uint8)
    outputs = {}
    for mode in ("dual", "six"):
        collector = AutoOpenStandardEpisodeCollector.__new__(AutoOpenStandardEpisodeCollector)
        collector.seed, collector.env_id, collector._fisheye_borders = 5, 71, {}
        calls = []

        def effective(name, env_id):
            calls.append((name, env_id))
            return spec

        collector.runner = SimpleNamespace(
            rgb_camera_manager=SimpleNamespace(camera_spec_for_env=effective)
        )
        frames = {name: image for name in camera_streams_for_mode(mode)}
        outputs[mode] = collector._augment_data_fisheye_borders(frames)
        transformed = transform_collection_images(outputs[mode], mode)
        assert len(transformed) == len(frames)
        assert len(calls) == len(frames)
        collector._augment_data_fisheye_borders(frames)
        assert len(calls) == len(frames)  # One texture/calibration lookup per episode.
        for name, output in outputs[mode].items():
            if "diagnostic" in name:
                np.testing.assert_array_equal(output, image)
            else:
                assert output[0, 0].max() <= 80
                np.testing.assert_array_equal(output[240, 320], image[240, 320])


def test_manager_uses_episode_environment_not_reused_render_slot():
    path = Path(__file__).resolve().parents[1] / (
        "third_party/visual_wholebody_lowlevel/legged_gym/scripts/rgb_camera_debug.py"
    )
    module_spec = importlib.util.spec_from_file_location("border_camera_manager", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    manager = module.SimRgbCameraManager.__new__(module.SimRgbCameraManager)
    a, b = {**SPEC, "name": "wrist"}, {**SPEC, "name": "wrist", "cx": 20}
    manager._episode_specs = {7: [a], 71: [b]}
    manager._camera_specs = {"wrist": a}
    assert manager.camera_spec_for_env("wrist", 71) is b
    assert manager.camera_spec_for_env("wrist", 7) is a
    manager._episode_specs = {}
    assert manager.camera_spec_for_env("wrist", 71) is a


@pytest.mark.parametrize("mode", ["dual", "six"])
def test_capture_before_rotation_and_finish_persists_actual_parameters(mode):
    from gr00t.rl.data_collection.auto_open_collector import AutoOpenStandardEpisodeCollector
    from gr00t.rl.data_collection.auto_open_standard import (
        camera_streams_for_mode,
        transform_collection_images,
    )

    spec = {
        **SPEC,
        "width": 640,
        "height": 480,
        "valid_circle_mask": {"center": [310, 235], "radius": 250, "feather": 2},
    }
    image = np.random.default_rng(3).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    collector = AutoOpenStandardEpisodeCollector.__new__(AutoOpenStandardEpisodeCollector)
    collector.seed, collector.env_id, collector.episode_index = 19, 7, 0
    collector._fisheye_borders = {}
    collector.runner = SimpleNamespace(
        rgb_camera_manager=SimpleNamespace(camera_spec_for_env=lambda name, env_id: spec)
    )
    collector.camera_streams = camera_streams_for_mode(mode)
    collector.camera_mode, collector.capture_images = mode, True
    collector._pending, collector.video_staging_root = None, None
    collector.states, collector.actions, collector.contacts = [], [], []
    collector.video_encoders, collector.events = {}, {}
    collector.images, collector.audit, collector.timing = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(float),
    )
    collector._snapshot_state = lambda state: {"packed_state": np.zeros(58, np.float32)}
    collector._append_state_audit = lambda *args: None
    frames = {name: image for name in collector.camera_streams}
    collector._read_current_images = lambda: frames
    collector.capture_before_step(None, np.zeros(25), stage_name="TEST", begin_contact_tick=False)
    expected = transform_collection_images(
        {name: FisheyeBorderAugmenter(19, name, spec).apply(image) for name in frames}, mode
    )
    for name, expected_image in expected.items():
        np.testing.assert_array_equal(collector.images[name][0], expected_image)
    collector._pending = None
    episode = collector.finish(provenance={"test_marker": True})
    assert episode.provenance["test_marker"]
    records = episode.provenance["fisheye_border_augmentation"]
    assert set(records) == set(frames)
    for name in frames:
        assert records[name] == FisheyeBorderAugmenter(19, name, spec).metadata
    json.dumps(episode.provenance)
