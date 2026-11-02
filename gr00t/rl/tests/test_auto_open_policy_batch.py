import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

LOWLEVEL_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "visual_wholebody_lowlevel"
sys.path.insert(0, str(LOWLEVEL_ROOT))

from legged_gym.scripts.manip_loco_interface import ManipLoco_Policy  # noqa: E402
from legged_gym.utils.episode_rng import AUTO_OPEN_POLICY_BLOCK_SIZE  # noqa: E402


class _RecordingPolicy:
    def __init__(self):
        self.shape = None
        self.shapes = []

    def __call__(self, observations, *, hist_encoding):
        assert hist_encoding is True
        self.shape = tuple(observations.shape)
        self.shapes.append(self.shape)
        return observations[:, :3]


@pytest.mark.parametrize("debug", [False, True])
def test_step_trace_preserves_raw_network_values_before_simulation_mutates_buffers(debug):
    runner = ManipLoco_Policy.__new__(ManipLoco_Policy)
    runner.args = SimpleNamespace(debug_trace=debug)
    runner.timestamp = 1
    runner.obs = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    expected = runner.obs.clone()
    runner._policy_actions = lambda obs: obs[:, :3]
    runner.rgb_camera_manager = None
    runner._read_rgb_frames = lambda: {}
    runner.get_control_state = lambda: {}

    def simulate(actions):
        if debug:
            runner.env.debug_physics_substeps.extend(
                [tuple(torch.full((2, 19), float(j + i)) for j in range(4)) for i in range(4)]
            )
        runner.obs.fill_(-10)
        actions.fill_(99)
        return (torch.zeros_like(expected),)

    runner.env = SimpleNamespace(step=simulate)
    result = runner.step_once()
    if debug:
        assert torch.equal(result["state"]["lowlevel_policy_observation"], expected)
        assert torch.equal(result["state"]["lowlevel_policy_action"], expected[:, :3])
        assert result["state"]["physics_substep_torque"].shape == (2, 4, 19)
        assert torch.equal(
            result["state"]["physics_substep_torque"][0, :, 0], torch.arange(2.0, 6.0)
        )
    else:
        assert result["state"] == {}


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


def test_seeded_non_door_collection_uses_configured_surviving_rows():
    runner = ManipLoco_Policy.__new__(ManipLoco_Policy)
    runner.env_cfg = type(
        "Cfg", (), {"enable_doorman_scene": False, "doorman_episode_seeds": (1700, 1704)}
    )()
    runner.args = type("Args", (), {"seed": 1700, "episode_seeds": [1700, 1702, 1704]})()
    runner.policy = _RecordingPolicy()
    observations = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    assert torch.equal(runner._policy_actions(observations), observations[:, :3])
    assert runner.policy.shapes == [(AUTO_OPEN_POLICY_BLOCK_SIZE, 5)]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_seeded_non_door_neural_inference_is_independent_of_batch_size_and_order(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    network = (
        torch.nn.Sequential(torch.nn.Linear(128, 256), torch.nn.ELU(), torch.nn.Linear(256, 18))
        .to(device)
        .eval()
    )
    observations = torch.randn(3, 128, device=device)

    def infer(seeds, rows):
        runner = ManipLoco_Policy.__new__(ManipLoco_Policy)
        runner.env_cfg = type(
            "Cfg", (), {"enable_doorman_scene": False, "doorman_episode_seeds": seeds}
        )()
        runner.args = type("Args", (), {"seed": 0})()
        runner.policy = lambda value, **kwargs: network(value)
        with torch.inference_mode():
            return runner._policy_actions(observations[rows])

    mixed = infer((1700, 1704, 2000), [0, 1, 2])
    for index, seed in enumerate((1700, 1704, 2000)):
        assert torch.equal(infer((seed,), [index])[0], mixed[index])
    assert torch.equal(infer((2000, 1700, 1704), [2, 0, 1]), mixed[[2, 0, 1]])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="deployed CUDA policy")
def test_deployed_history_policy_has_identical_seeded_single_and_mixed_outputs():
    from legged_gym.utils.helpers import load_policy_from_checkpoint

    from gr00t.rl.auto_open.cli import _default_paths

    args = SimpleNamespace(
        ckpt_path=None,
        run_metadata_path=None,
        robot_usd_path=None,
        standard_dataset_dir=None,
        disable_rgb_cameras=True,
    )
    _default_paths(args)
    if not Path(args.ckpt_path).is_file():
        pytest.skip("deployed checkpoint unavailable")
    network = load_policy_from_checkpoint(args.ckpt_path, "cuda")
    actor = network.actor
    width = actor.num_prop + actor.num_priv + actor.num_hist * actor.num_prop
    values = torch.randn(3, width, device="cuda")

    def infer(seeds, rows):
        runner = ManipLoco_Policy.__new__(ManipLoco_Policy)
        runner.env_cfg = SimpleNamespace(enable_doorman_scene=False, doorman_episode_seeds=seeds)
        runner.args = SimpleNamespace(seed=0)
        runner.policy = network
        return runner._policy_actions(values[rows])

    with torch.inference_mode():
        mixed = infer((1700, 1704, 2000), [0, 1, 2])
        for index, seed in enumerate((1700, 1704, 2000)):
            assert torch.equal(infer((seed,), [index])[0], mixed[index])
        assert torch.equal(infer((2000, 1704, 1700), [2, 1, 0]), mixed[[2, 1, 0]])
        native_single = network(values[:1], hist_encoding=True)[0]
        native_mixed = network(values, hist_encoding=True)[0]
    print(
        {
            "deployed_native_batch_max_action_difference": float(
                (native_single - native_mixed).abs().max()
            ),
            "fixed_seed_batch_action_difference": 0.0,
        }
    )
