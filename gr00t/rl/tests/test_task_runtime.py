"""Every task shares batch ownership, independent termination and policy stepping."""

from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.task_runtime import BatchedTaskLoop, BatchIdentity, reset_task_rows


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_partial_reset_keeps_partner_state_and_policy_history(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU reset")
    count = 4
    state = torch.arange(count * 3, device=device).reshape(count, 3).float() + 1
    env = SimpleNamespace(num_envs=count, device=device)
    env.obs_history_buf = torch.arange(count * 30, device=device).reshape(count, 10, 3).float()
    actions = state.clone()
    observations = state.clone()
    policy = SimpleNamespace(env=env, obs=observations)
    events, reset_calls = [], []

    def reset(rows):
        events.append("reset")
        reset_calls.append(rows.clone())
        state[rows], actions[rows], env.obs_history_buf[rows] = 0, 0, 0

    def observe():
        events.append("observe")
        env.obs_history_buf.add_(42)
        observations.copy_(state)
        return observations

    env._reset_idx = reset
    env.scene = SimpleNamespace(write_data_to_sim=lambda: events.append("write"))
    env.sim = SimpleNamespace(forward=lambda: events.append("forward"))
    env.synchronize_rgb_cameras_after_reset = lambda: events.append("cameras")
    env.get_observations = observe
    before = state.clone(), actions.clone(), env.obs_history_buf.clone(), observations.clone()
    rows = torch.tensor([1, 3], device=device)

    def forbidden(*args, **kwargs):
        raise AssertionError("Batch reset must not copy tensors to the host")

    with monkeypatch.context() as scope:
        for name in ("cpu", "item", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        reset_task_rows(policy, rows)
    assert events == ["reset", "write", "forward", "cameras", "observe"]
    assert len(reset_calls) == 1
    torch.testing.assert_close(reset_calls[0], rows)
    keep = torch.tensor([0, 2], device=device)
    for old, new in zip(before, (state, actions, env.obs_history_buf, policy.obs), strict=True):
        torch.testing.assert_close(old[keep], new[keep], rtol=0, atol=0)
    assert state[rows].eq(0).all() and actions[rows].eq(0).all()
    assert env.obs_history_buf[rows].eq(42).all()
    assert policy.obs[rows].eq(0).all()


def test_full_and_empty_reset_keep_existing_public_reset_path():
    calls = []
    env = SimpleNamespace(num_envs=2, device="cpu")
    env.reset = lambda: (calls.append("reset") or {"policy": torch.ones(2, 3)}, {})
    policy = SimpleNamespace(env=env)
    reset_task_rows(policy, torch.empty(0, dtype=torch.long))
    assert not calls
    reset_task_rows(policy, torch.arange(2))
    assert calls == ["reset"] and policy.obs.eq(1).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_payload_reset_does_not_restart_robot_or_observation_history(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU reset")
    events = []
    env = SimpleNamespace(num_envs=3, device=device)
    env.reset_payload_rows = lambda rows: events.append(("payload", rows.clone()))
    env._reset_idx = lambda rows: pytest.fail("Payload reset must not reset its robot")
    env.reset = lambda: pytest.fail("Payload reset must not reset its robot")
    env.scene = SimpleNamespace(write_data_to_sim=lambda: events.append(("write", None)))
    env.sim = SimpleNamespace(forward=lambda: events.append(("forward", None)))
    env.synchronize_rgb_cameras_after_reset = lambda: events.append(("cameras", None))
    env.get_observations = lambda: pytest.fail("No physics or robot state changed")
    policy = SimpleNamespace(env=env, obs=torch.randn(3, 4, device=device))
    before = policy.obs.clone()
    rows = torch.tensor([2, 0], device=device)
    preserve = torch.ones(3, device=device, dtype=torch.bool)
    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, lambda *a, **k: pytest.fail("GPU reset host read"))
        reset_task_rows(policy, rows, preserve_robot=preserve)
    assert [event for event, _ in events] == ["payload", "write", "forward", "cameras"]
    torch.testing.assert_close(events[0][1], rows)
    torch.testing.assert_close(policy.obs, before, rtol=0, atol=0)


def runner(count):
    env = SimpleNamespace(num_envs=count, device="cpu", num_actions=18)
    calls = []
    env._clip_teleop_targets = lambda: calls.append("clip")
    env._update_effective_teleop_inputs = lambda **kw: calls.append("prepare")

    def step():
        calls.append("physics")
        return {"low_level_action": torch.full((count, 18), float(len(calls)))}

    return SimpleNamespace(env=env, step_once=step), calls


def test_rank_partition_matches_existing_door_strided_assignment():
    ranks = BatchIdentity.partition(range(1000), range(4))
    assert [len(seeds) for _, seeds in ranks] == [250] * 4
    assert ranks[2] == (2, tuple(range(2, 1000, 4)))
    assert sorted(seed for _, seeds in ranks for seed in seeds) == list(range(1000))


def test_asset_affinity_preserves_400_seeds_and_100_simultaneous_rows_per_rank():
    metadata = {
        str(seed): {"key": f"scene_{seed % 20}", "weight": 1000 + seed % 20} for seed in range(400)
    }
    assignment = BatchIdentity.partition(range(400), range(4), affinity=metadata)
    assert [len(seeds) for _, seeds in assignment] == [100] * 4
    assert sorted(seed for _, seeds in assignment for seed in seeds) == list(range(400))
    assert assignment == BatchIdentity.partition(range(400), range(4), affinity=metadata)
    assert sum(len({metadata[str(seed)]["key"] for seed in seeds}) for _, seeds in assignment) == 20


def test_asset_affinity_splits_large_groups_and_keeps_uneven_rank_sizes():
    metadata = {str(seed): {"key": "shared_scene", "weight": 5} for seed in range(11)}
    assignment = BatchIdentity.partition(range(11), [2, 3, 0], affinity=metadata)
    assert [len(seeds) for _, seeds in assignment] == [4, 4, 3]
    assert sorted(seed for _, seeds in assignment for seed in seeds) == list(range(11))


@pytest.mark.parametrize(
    "metadata",
    [
        [],
        {},
        {"0": {"key": "room", "weight": float("nan")}},
        {"0": {"key": "room", "weight": -1}},
        {"0": {"key": "", "weight": 1}},
        {"0": {"key": "room", "weight": True}},
    ],
)
def test_invalid_asset_affinity_is_rejected_before_launch(metadata):
    with pytest.raises(ValueError):
        BatchIdentity.partition([0], [0], affinity=metadata)


def test_asset_affinity_rejects_inconsistent_weights_for_same_asset():
    metadata = {"0": {"key": "room", "weight": 1}, "1": {"key": "room", "weight": 2}}
    with pytest.raises(ValueError, match="inconsistent"):
        BatchIdentity.partition([0, 1], [0], affinity=metadata)


def test_terminal_rows_do_not_stop_other_environments():
    policy, calls = runner(3)
    loop = BatchedTaskLoop(
        policy, [1700, 1701, 1702], max_steps=8, success_stage=9, failure_stage=10
    )
    stage = torch.tensor([1, 1, 1])
    buffers = SimpleNamespace(flush=lambda env: calls.append("flush"))
    for step, active in loop.ticks(stage):
        assert (
            active.tolist() == [[True, True, True], [False, True, True], [False, False, True]][step]
        )
        loop.publish(buffers)
        loop.step()
        stage[step] = 9 if step != 1 else 10
    assert loop.step_counts.tolist() == [1, 2, 3]
    assert loop.physics_steps == 3
    assert calls == ["flush", "clip", "prepare", "physics"] * 3
    assert torch.isfinite(loop.first_actions).all()


def test_duplicate_physics_or_missing_physics_is_rejected():
    policy, _ = runner(1)
    loop = BatchedTaskLoop(policy, [1], max_steps=2, success_stage=9, failure_stage=10)
    with pytest.raises(RuntimeError, match="exactly once"):
        loop.step()
    ticks = loop.ticks(torch.tensor([1]))
    next(ticks)
    with pytest.raises(RuntimeError, match="without"):
        next(ticks)


@pytest.mark.parametrize("seeds", [[], [1, 1], [-1]])
def test_invalid_batch_seeds_rejected(seeds):
    with pytest.raises(ValueError):
        BatchIdentity(tuple(seeds))


def test_shared_stage_engine_orders_scene_capture_control_acceptance_and_planning():
    policy, calls = runner(2)
    frame = {"value": 0}

    def step():
        calls.append("physics")
        frame["value"] += 1
        return {"low_level_action": torch.zeros(2, 18), "state": dict(frame)}

    policy.step_once = step
    stage = torch.ones(2, dtype=torch.long)

    def command(state, buffers, *, metric):
        assert metric == state["value"]
        calls.append("command")

    def advance(state, *, metric):
        assert metric == state["value"]
        calls.append("advance")
        stage[state["value"] - 1] = 9
        return {"success": torch.tensor([state["value"] - 1])}

    fsm = SimpleNamespace(stage=stage, command=command, advance=advance)
    planner = SimpleNamespace(plan_events=lambda *args: calls.append("plan"))
    buffers = SimpleNamespace(flush=lambda env: calls.append("flush"))
    loop = BatchedTaskLoop(policy, [1700, 1701], max_steps=5, success_stage=9, failure_stage=10)
    timing = {
        name: 0.0
        for name in ("command_s", "teleop_prepare_s", "policy_physics_s", "advance_s", "planning_s")
    }
    frames = list(
        loop.run_stages(
            fsm,
            planner,
            frame,
            buffers,
            timing=timing,
            geometry=lambda state: {"metric": state["value"]},
            capture_before=lambda *args: calls.append("capture_before"),
            capture_after=lambda *args: calls.append("capture_after"),
            report=lambda *args: calls.append("report"),
        )
    )
    assert len(frames) == 2
    assert loop.step_counts.tolist() == [1, 2]
    assert (
        calls
        == [
            "command",
            "flush",
            "clip",
            "prepare",
            "capture_before",
            "physics",
            "capture_after",
            "advance",
            "report",
            "plan",
        ]
        * 2
    )
