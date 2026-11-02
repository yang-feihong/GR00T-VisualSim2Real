"""Reset policy history independently of other environments' admission retries."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("enabled", [False, True])
def test_reset_clears_only_requested_rows_and_is_wired_to_shared_reset(device, enabled):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    source = (
        Path(__file__).parents[1]
        / "third_party/visual_wholebody_lowlevel/legged_gym/envs/base/legged_robot.py"
    )
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    scope = {}
    exec(
        compile(
            ast.Module(body=[methods["_reset_policy_history"]], type_ignores=[]),
            str(source),
            "exec",
        ),
        scope,
    )
    names = ("actions", "last_actions", "action_history_buf", "obs_history_buf", "last_dof_vel")
    env = SimpleNamespace(**{name: torch.ones(3, 4, device=device) for name in names})
    env.reset_policy_history_on_reset = enabled
    env.dof_pos = torch.full((3, 4), 2.0, device=device)
    scope["_reset_policy_history"](env, torch.tensor([0, 2], device=device))
    for name in names:
        assert getattr(env, name)[[0, 2]].eq(0 if enabled else 1).all()
        assert getattr(env, name)[1].eq(1).all()
    assert env.dof_pos.eq(2).all()
    reset_source = ast.unparse(methods["_reset_idx"])
    assert "self._reset_policy_history(env_ids)" in reset_source
    defaults = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in cls.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "reset_policy_history_on_reset"
    }
    assert defaults["reset_policy_history_on_reset"] is False
    pick_source = (Path(__file__).parents[1] / "pick/environment.py").read_text()
    assert "reset_policy_history_on_reset = True" in pick_source
