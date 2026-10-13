import sys
from pathlib import Path

import torch


LOWLEVEL_ROOT = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "visual_wholebody_lowlevel"
)
sys.path.insert(0, str(LOWLEVEL_ROOT))

from legged_gym.scripts.manip_loco_interface import ManipLoco_Policy
from legged_gym.utils.episode_rng import AUTO_OPEN_POLICY_BLOCK_SIZE


class _RecordingPolicy:
    def __init__(self):
        self.shape = None
        self.shapes = []

    def __call__(self, observations, *, hist_encoding):
        assert hist_encoding is True
        self.shape = tuple(observations.shape)
        self.shapes.append(self.shape)
        return observations[:, :3]


def test_auto_open_policy_uses_fixed_inference_batch_and_returns_real_rows():
    runner = ManipLoco_Policy.__new__(ManipLoco_Policy)
    runner.env_cfg = type("Cfg", (), {"enable_doorman_scene": True})()
    runner.args = type("Args", (), {"seed": 1700})()
    runner.policy = _RecordingPolicy()
    observations = torch.arange(10, dtype=torch.float32).reshape(2, 5)

    actions = runner._policy_actions(observations)

    assert runner.policy.shape == (AUTO_OPEN_POLICY_BLOCK_SIZE, 5)
    assert torch.equal(actions, observations[:, :3])


def test_non_doorman_policy_keeps_native_batch_shape():
    runner = ManipLoco_Policy.__new__(ManipLoco_Policy)
    runner.env_cfg = type("Cfg", (), {"enable_doorman_scene": False})()
    runner.args = type("Args", (), {"seed": 1700})()
    runner.policy = _RecordingPolicy()
    observations = torch.zeros((3, 5), dtype=torch.float32)

    runner._policy_actions(observations)

    assert runner.policy.shape == (3, 5)


def test_auto_open_policy_supports_more_than_one_compute_block():
    runner = ManipLoco_Policy.__new__(ManipLoco_Policy)
    runner.env_cfg = type("Cfg", (), {"enable_doorman_scene": True})()
    runner.args = type("Args", (), {"seed": 1700})()
    runner.policy = _RecordingPolicy()
    observations = torch.arange(5000, dtype=torch.float32).reshape(1000, 5)

    actions = runner._policy_actions(observations)

    assert runner.policy.shapes == [(AUTO_OPEN_POLICY_BLOCK_SIZE, 5)] * 4
    assert torch.equal(actions, observations[:, :3])
