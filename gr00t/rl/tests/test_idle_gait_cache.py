"""Idle gait caching preserves source gait dynamics, outputs and reset transitions."""

import ast
import copy
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("pattern", ["adaptive_trot", "fixed_trot"])
def test_idle_gait_cache_matches_uncached_source_and_resets(device, pattern):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU gait state")
    source = Path(__file__).resolve().parents[1] / (
        "third_party/visual_wholebody_lowlevel/legged_gym/envs/base/legged_robot.py"
    )
    tree = ast.parse(source.read_text())
    parent = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    names = {
        "_reset_gait_state",
        "_set_fixed_trot_stopped_state",
        "_uniform_to_gait_pair_phases",
        "_start_gait_transitions",
        "_advance_gait_transitions",
        "_stop_gait_at_shared_stance",
        "_advance_constant_gait_frequency",
        "_refresh_gait_pair_phases",
        "_step_contact_targets",
    }
    methods = [copy.deepcopy(node) for node in parent.body if getattr(node, "name", None) in names]
    assert len(methods) == len(names)
    cls = ast.ClassDef(name="Gait", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.Module(body=[cls], type_ignores=[])
    namespace = {"torch": torch, "np": np, "time": time}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    Cached = namespace["Gait"]
    oracle = copy.deepcopy(module)
    step = next(node for node in oracle.body[0].body if node.name == "_step_contact_targets")
    guards = [
        node
        for node in step.body
        if isinstance(node, ast.If) and "idle_cache[0] == signature" in ast.unparse(node)
    ]
    assert len(guards) == 1
    step.body.remove(guards[0])
    exec(compile(ast.fix_missing_locations(oracle), "uncached_gait", "exec"), namespace)
    Uncached = namespace["Gait"]

    def make(cls):
        obj = cls()
        obj.device, obj.dt = device, 0.02
        obj.cfg = SimpleNamespace(
            env=SimpleNamespace(
                observe_gait_commands=True,
                gait_pattern=pattern,
                fixed_trot_frequency=1.5,
                gait_transition_duration_s=0.4,
            ),
            asset=SimpleNamespace(policy_foot_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
            rewards=SimpleNamespace(kappa_gait_probs=0.07),
        )
        for name in (
            "gait_frequencies",
            "gait_indices",
            "gait_stance_durations",
            "gait_transition_elapsed_s",
            "gait_transition_start_frequency",
            "gait_transition_target_frequency",
            "gait_transition_start_stance_duration",
            "gait_transition_target_stance_duration",
            "gait_requested_frequencies",
        ):
            setattr(obj, name, torch.zeros(3, device=device))
        obj.commands = torch.zeros(3, 3, device=device)
        obj.last_velocity_command = obj.commands.clone()
        obj.gait_frequency_commands = obj.commands.clone()
        obj.gait_transition_active = torch.zeros(3, device=device, dtype=torch.bool)
        obj.velocity_command_changed = obj.gait_transition_active.clone()
        obj.gait_pair_phases = torch.zeros(3, 2, device=device)
        for name in (
            "foot_indices",
            "clock_inputs",
            "doubletime_clock_inputs",
            "halftime_clock_inputs",
            "desired_contact_states",
        ):
            setattr(obj, name, torch.zeros(3, 4, device=device))
        obj._profile_record = lambda *args: None
        obj.get_gait_stance_duration = lambda frequency: torch.where(frequency > 0, 0.7, 1.0)
        obj._get_gait_frequency_targets = lambda: torch.where(
            obj.commands.norm(dim=-1) > 0.15, 1.5, 0.0
        )
        obj._reset_gait_state(torch.arange(3, device=device))
        obj.refresh_calls = 0
        refresh = obj._refresh_gait_pair_phases

        def counted():
            obj.refresh_calls += 1
            refresh()

        obj._refresh_gait_pair_phases = counted
        return obj

    cached, uncached = make(Cached), make(Uncached)
    tensors = [name for name, value in vars(cached).items() if torch.is_tensor(value)]
    for frame in range(240):
        for obj in (cached, uncached):
            if frame == 15:
                obj.gait_indices[1] += 0.125
            if frame == 25:
                obj._reset_gait_state(torch.tensor([1], device=device))
            if frame == 40:
                obj.commands[:, 0] = obj.commands.new_tensor([0.3, 0, -0.3])
            if frame == 70:
                obj.cfg.rewards.kappa_gait_probs = 0.09
            if frame == 90:
                obj.commands.zero_()
            obj._step_contact_targets()
        for name in tensors:
            torch.testing.assert_close(
                getattr(cached, name),
                getattr(uncached, name),
                rtol=0,
                atol=0,
                equal_nan=True,
                msg=lambda text: f"{frame}: {name}: {text}",
            )
    assert cached.refresh_calls < uncached.refresh_calls / 2
