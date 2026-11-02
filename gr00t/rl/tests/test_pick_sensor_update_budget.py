"""Current-only contact sensors keep final substep values without eager history work."""

import ast
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.skipif(not os.environ.get("PICK_SENSOR_BASELINE"), reason="Recorded probes required")
def test_recorded_current_only_sensor_equivalence(tmp_path):
    from gr00t.rl.auto_open.tensor_metadata import to_host

    baseline = Path(os.environ["PICK_SENSOR_BASELINE"])
    candidate = Path(os.environ["PICK_SENSOR_CANDIDATE"])
    before = json.loads((baseline / "summary.json").read_text())
    after = json.loads((candidate / "summary.json").read_text())
    for key in (
        "seeds", "scene_ids", "asset_names", "initial_root_states", "initial_object_states"
    ):
        assert before[key] == after[key], key
    old = torch.load(baseline / "stability_history.pt", map_location="cuda", weights_only=True)
    new = torch.load(candidate / "stability_history.pt", map_location="cuda", weights_only=True)
    records = {}
    for key in ("ready", "root_states", "dof_pos", "object_state", "distractor_states"):
        assert old[key].shape == new[key].shape
        equal = old[key] == new[key]
        if old[key].is_floating_point():
            equal |= old[key].isnan() & new[key].isnan()
        frames = equal.reshape(equal.shape[0], -1).all(-1)
        indices = torch.arange(len(frames), device=frames.device)
        records[key] = {
            "bitwise_equal": equal.all(),
            "first_different_frame": indices.masked_fill(frames, len(frames)).amin(),
        }
    report = {
        "physics_simulated_here": False,
        "identical_initial_scenes": True,
        "baseline_policy_physics_s": before["policy_physics_s"],
        "candidate_policy_physics_s": after["policy_physics_s"],
        "physics_state_comparison": to_host(records),
    }
    (tmp_path / "sensor_refresh_equivalence.json").write_text(json.dumps(report, indent=2))
    assert all(record["bitwise_equal"] for record in report["physics_state_comparison"].values())


def test_pick_contact_consumers_request_no_unused_sensor_history():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "pick/environment.py").read_text())
    configs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ContactSensorCfg"
    ]
    assert len(configs) == 2
    for config in configs:
        values = {key.arg: key.value for key in config.keywords}
        assert ast.literal_eval(values["history_length"]) == 0
        assert ast.literal_eval(values["update_period"]) == 0
    for relative in (
        "pick/environment.py",
        "pick/collection.py",
        "data_collection/contact_audit.py",
    ):
        parsed = ast.parse((root / relative).read_text())
        assert not any(
            isinstance(node, ast.Attribute)
            and node.attr in {"force_matrix_w_history", "net_forces_w_history"}
            for node in ast.walk(parsed)
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sdk_lazy_refresh_keeps_the_final_physical_sample_and_force_dt(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU sensor scheduling")
    spec = importlib.util.find_spec("isaaclab")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("Isaac Lab SDK source required")
    sensors = Path(next(iter(spec.submodule_search_locations))) / "sensors"
    tree = ast.parse((sensors / "sensor_base.py").read_text())
    sensor = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SensorBase"
    )
    update = next(
        node for node in sensor.body if isinstance(node, ast.FunctionDef) and node.name == "update"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[update], type_ignores=[]), "sdk_sensor_update", "exec"), namespace
    )

    class Samples:
        def __init__(self, history):
            self.cfg = SimpleNamespace(update_period=0, history_length=history)
            self._is_visualizing = False
            self._timestamp = torch.zeros(2, device=device)
            self._timestamp_last_update = self._timestamp.clone()
            self._is_outdated = torch.zeros(2, device=device, dtype=torch.bool)
            self.actual = torch.zeros(2, 3, device=device)
            self.snapshot = self.actual.clone()
            self.calls = 0

        def _update_outdated_buffers(self):
            self.snapshot.copy_(self.actual)
            self._timestamp_last_update.copy_(self._timestamp)
            self._is_outdated.zero_()
            self.calls += 1

    eager, lazy = Samples(1), Samples(0)
    for step in range(4):
        for item in (eager, lazy):
            item.actual.fill_(step + 1)
            namespace["update"](item, 0.005)
    assert eager.calls == 4 and lazy.calls == 0
    lazy._update_outdated_buffers()
    assert lazy.calls == 1
    torch.testing.assert_close(eager.snapshot, lazy.snapshot, rtol=0, atol=0)
    contact = ast.parse((sensors / "contact_sensor/contact_sensor.py").read_text())
    force_calls = [
        node
        for node in ast.walk(contact)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr
        in {
            "get_net_contact_forces",
            "get_contact_force_matrix",
            "get_contact_data",
            "get_friction_data",
        }
    ]
    assert len(force_calls) == 4
    for call in force_calls:
        dt = next(keyword.value for keyword in call.keywords if keyword.arg == "dt")
        assert isinstance(dt, ast.Attribute) and dt.attr == "_sim_physics_dt"
