import ast
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gr00t.rl.data_collection.auto_open_collector import AutoOpenStandardEpisodeCollector


def test_door_audit_preserves_network_trace_and_action_frame_alignment(tmp_path):
    import pyarrow.parquet as pq

    from gr00t.rl.data_collection.auto_open_writer import _write_raw_audit

    collector = AutoOpenStandardEpisodeCollector.__new__(AutoOpenStandardEpisodeCollector)
    collector.audit = defaultdict(list)
    collector.gripper_index = 0
    collector.controller = SimpleNamespace(hinge_idx=0, handle_joint_idx=1, latch_joint_idx=2)
    state = {
        "root_states": torch.zeros(1, 13),
        "dof_pos": torch.zeros(1, 19),
        "dof_vel": torch.zeros(1, 19),
        "ee_pos": torch.zeros(1, 3),
        "ee_orn": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "door_joint_position": torch.zeros(1, 3),
        "door_joint_velocity": torch.zeros(1, 3),
        "lowlevel_policy_observation": torch.arange(128)[None],
        "lowlevel_policy_action": torch.arange(18)[None],
        "physics_substep_torque": torch.arange(76).reshape(1, 4, 19),
    }
    collector._append_debug_post_state_audit(state, 126)
    state["lowlevel_policy_observation"].zero_()
    state["lowlevel_policy_action"].zero_()
    np.testing.assert_array_equal(collector.audit["lowlevel_policy_observation"][0], np.arange(128))
    np.testing.assert_array_equal(collector.audit["lowlevel_policy_action"][0], np.arange(18))
    assert collector.audit["post_state_frame_index"] == [126]
    path = tmp_path / "raw.parquet"
    _write_raw_audit(path, collector.audit, 1)
    restored = pq.read_table(path).to_pydict()
    assert restored["lowlevel_policy_observation"] == [list(range(128))]
    assert restored["lowlevel_policy_action"] == [list(range(18))]
    assert restored["post_state_frame_index"] == [126]
    assert restored["physics_substep_torque"] == [list(range(76))]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_physics_substep_capture_preserves_gpu_values_and_is_disabled_without_debug(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    path = Path(__file__).parents[1] / (
        "third_party/visual_wholebody_lowlevel/legged_gym/envs/manip_loco/manip_loco.py"
    )
    cls = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef))
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    namespace = {}
    exec(
        compile(
            ast.Module(body=[methods["_capture_debug_physics_substep"]], type_ignores=[]),
            str(path),
            "exec",
        ),
        namespace,
    )
    capture = namespace["_capture_debug_physics_substep"]
    assert "self._capture_debug_physics_substep(pos_targets)" in ast.unparse(
        methods["_apply_action"]
    )
    env = SimpleNamespace(
        dof_pos=torch.full((2, 19), 1.0, device=device),
        dof_vel=torch.full((2, 19), 2.0, device=device),
        torques=torch.full((2, 19), 3.0, device=device),
    )
    target = torch.full((2, 19), 4.0, device=device)
    capture(env, target)
    assert not hasattr(env, "debug_physics_substeps")
    env.debug_physics_substeps = []
    capture(env, target)
    for value in (env.dof_pos, env.dof_vel, env.torques, target):
        value.zero_()
    for index, value in enumerate(env.debug_physics_substeps[0]):
        assert value.device == target.device
        assert value.eq(index + 1).all()
