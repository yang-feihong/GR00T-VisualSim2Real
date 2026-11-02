"""Admission diagnostics distinguish physics-first divergence from controller errors."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gr00t.rl.pick.collection import InitializationDebugRecorder
from gr00t.rl.pick.inspect_run import compare_initialization_probes


@pytest.mark.parametrize("controller_first", [False, True])
def test_first_substep_comparison_does_not_excuse_an_earlier_control_difference(
    tmp_path, controller_first
):
    paths = [tmp_path / "current", tmp_path / "reference"]
    for index, path in enumerate(paths):
        path.mkdir()
        (path / "stability_probe.json").write_text(json.dumps({"seeds": [42]}))
        record = {
            "initial": {
                key: torch.zeros(1, 3)
                for key in ("root_states", "dof_pos", "dof_vel", "object_state")
            },
            "lowlevel_policy_observation": torch.zeros(3, 1, 4),
            "lowlevel_policy_action": torch.zeros(3, 1, 2),
            **{
                key: torch.zeros(3, 1, 4, 2)
                for key in (
                    "physics_substep_dof_pos",
                    "physics_substep_dof_vel",
                    "physics_substep_torque",
                    "physics_substep_position_target",
                )
            },
        }
        if index == 0:
            record["physics_substep_dof_vel"][0, 0, 1, 0] = 1e-6
            record["lowlevel_policy_action"][0 if controller_first else 1, 0, 0] = 1e-6
        torch.save(record, path / "stability_history.pt")
    result = compare_initialization_probes(*paths)["episodes"][0]
    assert result["origin"] == (
        "requires_further_analysis" if controller_first else "physics_substep_after_equal_inputs"
    )
    assert result["first_difference_by_field"]["physics_substep_dof_vel"] == {
        "step": 1,
        "substep": 1,
        "maximum_absolute_difference": pytest.approx(1e-6),
    }


def test_initialization_video_is_a_passive_throttled_observer(monkeypatch, tmp_path):
    from gr00t.rl.data_collection import auto_open_writer

    encoders, render_calls = [], []

    class Encoder:
        def __init__(self, path, shape):
            self.path, self.shape, self.frames, self.closed = path, shape, [], False
            encoders.append(self)

        def append(self, frame):
            assert frame.shape == self.shape
            self.frames.append(frame)

        def close(self):
            self.closed = True

    def render(rows, rig_names):
        render_calls.append((rows, rig_names))
        return {
            row: {name: (np.zeros((8, 8, 3), dtype=np.uint8), None) for name in rig_names}
            for row in rows
        }

    monkeypatch.setattr(auto_open_writer, "StreamingVideoEncoder", Encoder)
    recorder = InitializationDebugRecorder(
        SimpleNamespace(rgb_camera_manager=SimpleNamespace(render_images_batch=render)),
        [3, 7],
        tmp_path,
        "pick_basket",
    )
    recorder.capture(0, 0)
    recorder.capture(0, 1)
    recorder.capture(0, 10)
    recorder.close()
    assert len(render_calls) == 2 and len(encoders) == 2
    assert all(len(encoder.frames) == 2 and encoder.closed for encoder in encoders)
    assert (
        json.loads((tmp_path / "initialization_video_frames.json").read_text())["playback_speed"]
        == 10
    )
