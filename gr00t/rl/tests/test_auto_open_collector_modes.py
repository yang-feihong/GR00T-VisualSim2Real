from types import SimpleNamespace

import numpy as np

from gr00t.rl.data_collection.auto_open_collector import (
    AutoOpenStandardEpisodeCollector,
)


def _collector(*, joint_mode=True):
    collector = AutoOpenStandardEpisodeCollector.__new__(
        AutoOpenStandardEpisodeCollector
    )
    collector.env = SimpleNamespace(
        teleop_arm_joint_control_mask=np.asarray([joint_mode], dtype=bool)
    )
    collector.env_id = 0
    collector.controller = SimpleNamespace(
        middle_arm_joint_target=np.asarray(
            [-1.57, 1.60, -0.63, -1.10, 0.0, 0.0], dtype=np.float32
        )
    )
    collector.arm_indices = np.arange(6, dtype=np.int64)
    collector._middle_preset_interval = False
    collector._middle_preset_reset_active = False
    collector._middle_preset_reset_complete = False
    collector._middle_preset_stable_frames = 0
    return collector


def _snapshot(position, velocity=None):
    return {
        "dof_pos": np.asarray(position, dtype=np.float32),
        "dof_vel": np.zeros(6, dtype=np.float32)
        if velocity is None
        else np.asarray(velocity, dtype=np.float32),
        "arm_velocity": np.zeros(6, dtype=np.float32)
        if velocity is None
        else np.asarray(velocity, dtype=np.float32),
    }


def test_cartesian_control_is_active():
    collector = _collector(joint_mode=False)

    assert collector._mode_flags(_snapshot(np.zeros(6)), np.zeros(6)) == (
        False,
        False,
    )


def test_joint_hold_is_inactive():
    collector = _collector()
    hold_target = np.full(6, 0.25, dtype=np.float32)

    assert collector._mode_flags(_snapshot(hold_target), hold_target) == (True, False)


def test_middle_preset_motion_is_reset():
    collector = _collector()
    middle = collector.controller.middle_arm_joint_target

    assert collector._mode_flags(_snapshot(middle + 0.10), middle) == (False, True)
    assert collector._mode_flags(
        _snapshot(middle, velocity=np.full(6, 0.10)), middle
    ) == (False, True)


def test_settled_middle_preset_is_inactive():
    collector = _collector()
    middle = collector.controller.middle_arm_joint_target

    assert collector._mode_flags(_snapshot(middle + 0.01), middle) == (True, False)


def test_middle_preset_reset_completion_is_latched_across_velocity_noise():
    collector = _collector()
    middle = collector.controller.middle_arm_joint_target

    assert collector._mode_flags(_snapshot(middle + 0.10), middle) == (False, True)
    for _ in range(9):
        assert collector._mode_flags(_snapshot(middle), middle) == (False, True)
    assert collector._mode_flags(_snapshot(middle), middle) == (True, False)
    assert collector._mode_flags(
        _snapshot(middle, velocity=np.full(6, 0.20)), middle
    ) == (True, False)
