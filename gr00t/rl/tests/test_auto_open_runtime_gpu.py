import ast
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "third_party" / "visual_wholebody_lowlevel")
)

from gr00t.rl.envs.door.b2z1_reset_geometry import b2z1_gripper_pose_and_jacobian
from gr00t.rl.third_party.visual_wholebody_lowlevel.legged_gym.utils.batch_invariant_math import (
    fixed_normalize,
)
from gr00t.rl.data_collection.auto_open_collector import batched_target_height_invariant_poses


# Load the pure helpers without importing the simulation/planner entry stack.
_RUNTIME = Path(__file__).resolve().parents[1] / "auto_open" / "runtime.py"
_HELPERS = {"_target_height_invariant_poses_gpu", "_write_stage_progress"}
_tree = ast.parse(_RUNTIME.read_text())
_module = ast.Module(
    body=[
        node for node in _tree.body if isinstance(node, ast.FunctionDef) and node.name in _HELPERS
    ],
    type_ignores=[],
)
exec(compile(_module, str(_RUNTIME), "exec"), globals())


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
def test_runtime_dataset_poses_match_offline_reference(device):
    generator = torch.Generator().manual_seed(52)
    roots = torch.randn(17, 13, generator=generator)
    arms = torch.randn(17, 6, generator=generator)
    expected = batched_target_height_invariant_poses(roots.numpy(), arms.numpy())
    actual = _target_height_invariant_poses_gpu(roots.to(device), arms.to(device))
    assert actual.device.type == device
    assert actual.dtype == torch.float32
    np.testing.assert_allclose(actual.cpu().numpy(), expected, atol=1.0e-6, rtol=1.0e-6)
    changed_velocities = roots.to(device).clone()
    changed_velocities[:, 7:13] = 1000.0
    torch.testing.assert_close(
        actual,
        _target_height_invariant_poses_gpu(changed_velocities, arms.to(device)),
        atol=0.0,
        rtol=0.0,
    )
    for index in (0, 8, 16):
        single = _target_height_invariant_poses_gpu(
            roots[index : index + 1].to(device), arms[index : index + 1].to(device)
        )
        torch.testing.assert_close(actual[index : index + 1], single, atol=1.0e-6, rtol=1.0e-6)


def test_progress_snapshot_stays_tensor_and_does_not_alias_stage(tmp_path):
    from enum import IntEnum
    from types import SimpleNamespace

    class Stage(IntEnum):
        APPROACH_HANDLE = 1
        SUCCESS = 2

    fsm = SimpleNamespace(
        stage=torch.tensor([1, 1, 2]),
        stage_steps=torch.tensor([3, 7, 0]),
        stage_limits=torch.tensor([0, 20, 0]),
    )
    snapshot = _write_stage_progress(tmp_path, Stage, fsm, 9, 1.25)
    torch.testing.assert_close(snapshot, fsm.stage)
    fsm.stage[0] = 2
    assert snapshot[0] == 1
    output = (tmp_path / "progress.txt").read_text()
    assert "stage=APPROACH_HANDLE count=2 stage_steps=3-7/20" in output
    assert "stage=SUCCESS count=1 stage_steps=0-0/terminal" in output
