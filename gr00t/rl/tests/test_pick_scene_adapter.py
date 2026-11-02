"""Reset identities survive reordering and scene-origin placement."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from gr00t.rl.pick.scene_adapter import batch_reset_tensors, create_pick_environment


def test_probe_accepts_prepared_batch_and_individual_episode_files(tmp_path):
    import json

    from gr00t.rl.pick.scene_adapter import load_probe_episodes

    batch = tmp_path / "prepared_scenes.json"
    single = tmp_path / "episode.json"
    batch.write_text(json.dumps([{"episode": {"seed": 7}}, {"episode": {"seed": 9}}]))
    single.write_text(json.dumps({"seed": 10}))
    assert [e["seed"] for e in load_probe_episodes([batch, single])] == [7, 9, 10]
    with pytest.raises(ValueError, match="distinct"):
        load_probe_episodes([batch, batch])


@pytest.mark.parametrize("component_profile", [False, True])
def test_stability_probe_dispatch_writes_traces_with_shared_reset_and_commands(
    monkeypatch, tmp_path, component_profile
):
    import argparse
    import json
    import math
    from pathlib import Path

    from gr00t.rl import task_runtime
    from gr00t.rl.auto_open import cli, execution
    from gr00t.rl.pick import scene_adapter as adapter

    episodes = [
        {
            "seed": seed,
            "task_type": "pick_basket",
            "scene_id": "fixture",
            "asset": {"name": "fixture"},
            "initial_arm_joints": [angle] * 6,
        }
        for seed, angle in ((7, 0.1), (9, 0.2))
    ]
    manifest = tmp_path / "episodes.json"
    manifest.write_text(json.dumps([{"episode": episode} for episode in episodes]))
    root = torch.zeros(2, 13)
    root[:, 2], root[:, 3] = 0.55, 1
    target = root.clone()
    target[:, 2] = 0.7
    state = {
        "root_states": root,
        "object_state": target,
        "distractor_states": torch.zeros(2, 0, 13),
        "dof_pos": torch.zeros(2, 7),
    }
    progress = {"steps": 0, "resets": 0, "observations": 0, "closed": False}
    profile_calls = []

    def profile():
        profile_calls.append(progress["steps"])
        return {"_steps": progress["steps"]}

    def observations():
        progress["observations"] += 1
        return torch.zeros(2, 18)

    def reset():
        progress["resets"] += 1
        return {"policy": observations()}, {}

    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        reset=reset,
        get_observations=observations,
        get_control_state=lambda: state,
        pop_step_profile_stats=profile,
        arm_joint_ids=torch.arange(6),
        scene=SimpleNamespace(env_origins=torch.zeros(2, 3)),
    )
    buffers = SimpleNamespace(
        base=torch.zeros(2, 3),
        arm_joint_mode=torch.zeros(2, dtype=torch.bool),
        arm_joint=torch.zeros(2, 6),
        gripper=torch.zeros(2),
    )
    expected = torch.tensor([episode["initial_arm_joints"] for episode in episodes])

    class Policy:
        def __init__(self, args):
            self.args = args
            self.init_env()

        def close(self):
            progress["closed"] = True

    class Loop:
        def __init__(self, runner, seeds, *, max_steps, **kwargs):
            self.maximum, self.physics_steps, self.policy_physics_s = max_steps, 0, 0.0
            self.publish_s = 0.25

        def ticks(self, stage):
            return ((step, None) for step in range(self.maximum))

        def publish(self, current):
            assert current is buffers and current.base.eq(0).all() and current.arm_joint_mode.all()
            torch.testing.assert_close(current.arm_joint, expected)
            torch.testing.assert_close(current.gripper, torch.full((2,), -math.pi / 3))

        def step(self):
            self.physics_steps += 1
            progress["steps"] = self.physics_steps
            return {"state": state}

    def parser():
        result = argparse.ArgumentParser()
        result.add_argument("--headless", action="store_true")
        result.add_argument("--max_steps", type=int, default=32)
        result.add_argument("--print_timing_breakdown", action="store_true")
        result.add_argument("--output_dir", type=Path, required=True)
        return result

    app_module = ModuleType("legged_gym.utils.isaaclab_app")
    app_module.launch_app = lambda args: SimpleNamespace(close=lambda: None)
    policy_module = ModuleType("legged_gym.scripts.manip_loco_interface")
    policy_module.ManipLoco_Policy, policy_module.load_policy = Policy, lambda *args: object()
    metadata_module = ModuleType("legged_gym.utils.run_metadata")
    metadata_module.checkpoint_model_path = lambda args: Path("fixture.pt")
    for module in (app_module, policy_module, metadata_module):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(cli, "_parser", parser)
    monkeypatch.setattr(cli, "_default_paths", lambda args: None)
    monkeypatch.setattr(adapter, "create_pick_environment", lambda args, source: (env, None))
    monkeypatch.setattr(task_runtime, "BatchedTaskLoop", Loop)
    monkeypatch.setattr(execution.TensorCommandBuffers, "from_env", lambda env: buffers)
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "--episodes",
            str(manifest),
            "--profile-stability",
            "--headless",
            "--output_dir",
            str(output),
        ]
        + (["--print_timing_breakdown"] if component_profile else []),
    )
    original_cpu = torch.Tensor.cpu

    def cpu_after_steps(value, *args, **kwargs):
        assert progress["steps"] == 32
        return original_cpu(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu_after_steps)
    adapter.main()
    report = json.loads((output / "stability_probe.json").read_text())
    assert report["pose_stable_first_step"] == [25, 25]
    assert report["first_poll_all_pose_stable_step"] == 32
    assert report["full_grasp_admission_evaluated"] is False
    assert progress == {"steps": 32, "resets": 1, "observations": 2, "closed": True}
    history = torch.load(output / "stability_history.pt", weights_only=True)
    assert history["root_states"].shape == (32, 2, 13)
    assert history["ready"].shape == (32, 2)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["command_publish_s"] == 0.25
    assert summary["environment_step_profile"] == ({"_steps": 32} if component_profile else None)
    assert profile_calls == ([0, 32] if component_profile else [])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_probe_stability_metrics_record_first_readiness_without_host_reads(monkeypatch, device):
    from gr00t.rl.pick.scene_adapter import stability_profile_metrics

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU metrics")
    ready = torch.zeros(16, 3, device=device, dtype=torch.bool)
    ready[1:, 0], ready[9:, 1], ready[15, 2] = True, True, True
    with monkeypatch.context() as scope:
        for name in ("cpu", "tolist", "item"):
            scope.setattr(
                torch.Tensor, name, lambda *args, **kwargs: pytest.fail("Metric host read")
            )
        result = stability_profile_metrics(ready)
    assert result["pose_stable_first_step"].tolist() == [2, 10, 16]
    assert result["pose_stable_frames_total"].tolist() == [15, 7, 1]
    assert result["first_all_pose_stable_step"] == 16
    assert result["first_poll_all_pose_stable_step"] == 16
    assert result["first_poll_80pct_pose_stable_step"] == 16
    ready[:, 2] = False
    result = stability_profile_metrics(ready)
    assert result["pose_stable_first_step"].tolist() == [2, 10, -1]
    assert result["first_poll_all_pose_stable_step"] == -1


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_basket_payload_reset_uses_measured_root_frame_without_origin_double_add(device):
    from gr00t.rl.pick.scene_adapter import basket_payload_reset_tensors

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU reset")
    roots = torch.zeros(2, 13, device=device)
    roots[:, :3] = roots.new_tensor([[100, -40, 0.51], [-20, 5, 0.54]])
    roots[:, 3:7] = roots.new_tensor([[2**-0.5, 0, 0, 2**-0.5], [1, 0, 0, 0]])
    origins = roots.new_tensor([[96, -32, 0], [-32, 0, 0]])
    descriptors = [
        {
            "basket_local_position": [0.2, 0, 0.15],
            "basket_local_yaw": 0,
            "object_position": [0, 0, 0],
        },
        {"object_position": [0, 0, -100]},
    ]
    reset = basket_payload_reset_tensors([descriptors], roots, origins)[0]
    torch.testing.assert_close(reset[:, :3], roots.new_tensor([[100, -39.8, 0.66], [-32, 0, -100]]))
    torch.testing.assert_close(reset[0, 3:7], roots[0, 3:7])
    assert reset[:, 7:].eq(0).all()


def test_pick_uses_sdk_in_memory_stage_without_changing_gpu_or_solver_configuration(monkeypatch):
    cfg = SimpleNamespace(
        sim=SimpleNamespace(device="cuda:0", create_stage_in_memory=False),
        scene=SimpleNamespace(),
        env=SimpleNamespace(),
        robot=SimpleNamespace(spawn=SimpleNamespace(rigid_props=SimpleNamespace())),
    )
    helpers = ModuleType("legged_gym.utils.helpers")
    helpers.build_env_cfg = lambda args, for_play: cfg
    environment = ModuleType("gr00t.rl.pick.environment")
    environment.PickEnvironment = lambda config, episodes: SimpleNamespace(
        cfg=config, episodes=episodes
    )
    monkeypatch.setitem(sys.modules, helpers.__name__, helpers)
    monkeypatch.setitem(sys.modules, environment.__name__, environment)
    episodes = [{"seed": 1}, {"seed": 7}]
    env, actual_cfg = create_pick_environment(SimpleNamespace(), episodes)
    assert actual_cfg is cfg and env.cfg is cfg and env.episodes is episodes
    assert cfg.sim.create_stage_in_memory
    assert cfg.sim.device == "cuda:0"
    assert not cfg.scene.replicate_physics
    assert cfg.robot.spawn.rigid_props.solver_position_iteration_count == 16
    assert cfg.robot.spawn.rigid_props.solver_velocity_iteration_count == 2


def test_distinct_initial_states_are_not_cloned_from_first_episode():
    episodes = [
        {
            "base_xy": [i, -i],
            "base_yaw": i / 4,
            "initial_arm_joints": [i / 10] * 6,
            "object_position": [i / 3, 1, 0.1 + i],
            "object_yaw": -i / 3,
        }
        for i in range(3)
    ]
    origins = torch.tensor([[0, 0, 0], [32, 0, 0], [0, 32, 0]], dtype=torch.float32)
    arm = torch.arange(6)
    gripper = torch.tensor([6])
    joints = torch.zeros(3, 7)
    all_rows = batch_reset_tensors(episodes, origins, joints, arm, gripper)
    for i in range(3):
        single = batch_reset_tensors(
            [episodes[i]], origins[i : i + 1], joints[i : i + 1], arm, gripper
        )
        torch.testing.assert_close(all_rows[0][i : i + 1], single[0])
        torch.testing.assert_close(all_rows[1][i : i + 1], single[1])
        torch.testing.assert_close(all_rows[2][0][i : i + 1], single[2][0])
