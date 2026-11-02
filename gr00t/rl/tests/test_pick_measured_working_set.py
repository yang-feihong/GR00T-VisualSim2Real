"""Compare fused working-set blocks to legacy blocks on recorded floor inputs."""

import ast
import inspect
import json
import os
import time
from pathlib import Path

import pytest
import torch

from gr00t.rl.tests.test_pick_recorded_batch_invariance import _recorded_batch


@pytest.mark.skipif(not os.environ.get("PICK_MEASURED_RECORD_ROOT"), reason="Records required")
def test_recorded_floor_working_set(tmp_path, monkeypatch):
    from gr00t.rl.pick.measured_commands import load_measured_commands
    from gr00t.rl.pick.measured_station import COMMAND_CATALOG
    from gr00t.rl.pick.measured_station_batch import select_measured_station_batch

    root = Path(os.environ["PICK_MEASURED_RECORD_ROOT"])
    count = min(8, len(list(root.glob("logs/rank*-control/seed_*/control_record.pt"))))
    state, destination, _, supports, mask, geometry, collision, _ = _recorded_batch(
        root, count=count
    )
    rows = torch.where(state["on_floor"])[0]
    assert len(rows), "Floor records required"
    catalogs = [COMMAND_CATALOG]
    if os.environ.get("PICK_MEASURED_CATALOG"):
        catalogs = [Path(os.environ["PICK_MEASURED_CATALOG"])]
    if os.environ.get("PICK_MEASURED_RAW_ROOT"):
        catalogs = [Path(os.environ["PICK_MEASURED_RAW_ROOT"]) / f"rank{rank}" for rank in range(2)]
    if os.environ.get("PICK_MEASURED_EXTRA_CATALOG"):
        catalogs.append(Path(os.environ["PICK_MEASURED_EXTRA_CATALOG"]))
    commands = load_measured_commands(
        catalogs,
        destination,
        maximum_position_error_m=float(os.environ.get("PICK_MEASURED_COMMAND_ERROR", "0.004")),
    )
    if os.environ.get("PICK_MEASURED_CAPTURE_CORRIDOR"):
        import gr00t.rl.pick.measured_station_batch as station_module

        original = station_module.station_corridors_clear
        captures = []

        def capture(roots, stations, view):
            original_visible = view.visible

            def visible(starts, ends, tolerance_m=0.01):
                hit, normal, distance = view.rays(starts, ends)
                blocked = distance < (ends - starts).norm(dim=-1) - tolerance_m
                indices = blocked.flatten(1).int().argmax(-1)
                row = torch.arange(len(indices), device=indices.device)
                captures.append(
                    {
                        "roots": roots.tolist(),
                        "stations": stations.tolist(),
                        "blocked": blocked.flatten(1).any(-1).tolist(),
                        "hit": hit.flatten(1, -2)[row, indices].tolist(),
                        "normal": normal.flatten(1, -2)[row, indices].tolist(),
                        "type": view.last_hit_types.flatten(1)[row, indices].tolist(),
                    }
                )
                return ~blocked

            view.visible = visible
            try:
                return original(roots, stations, view)
            finally:
                view.visible = original_visible
                (tmp_path / "corridor_hits.json").write_text(json.dumps(captures))

        monkeypatch.setattr(station_module, "station_corridors_clear", capture)

    class LegacyReference(ast.NodeTransformer):
        def visit_Assign(self, node):
            if any(
                isinstance(target, ast.Name) and target.id == "point_factor"
                for target in node.targets
            ):
                node.value = ast.Constant(17)
            return self.generic_visit(node)

    tree = LegacyReference().visit(ast.parse(inspect.getsource(select_measured_station_batch)))
    scope = dict(select_measured_station_batch.__globals__)
    exec(compile(ast.fix_missing_locations(tree), "exhaustive_measured_reference", "exec"), scope)
    values, report = [], []
    for name, function in (
        ("legacy_blocks", scope["select_measured_station_batch"]),
        ("fused_blocks", select_measured_station_batch),
    ):
        if os.environ.get("PICK_MEASURED_SKIP_LEGACY") and name == "legacy_blocks":
            continue
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = function(
            {key: value[rows] for key, value in state.items()},
            destination[rows],
            supports[rows],
            mask[rows],
            state["support_z"][rows],
            commands,
            collision,
            geometry.select(rows),
        )
        torch.cuda.synchronize()
        report.append(
            {
                "policy": name,
                "elapsed_s": time.perf_counter() - started,
                "seeds": state["episode_seed"][rows].tolist(),
                "valid": result[3].tolist(),
                "safe_counts": result[4]["scene_safe_candidate_count"].tolist(),
                "matched_counts": result[4]["matched_geometry_count"].tolist(),
                "open_counts": result[4]["open_clear_candidate_count"].tolist(),
                "close_counts": result[4]["closing_clear_candidate_count"].tolist(),
                "corridor_counts": result[4]["corridor_clear_candidate_count"].tolist(),
                "selected_station": result[0].tolist(),
                "selected_orientation": result[1].tolist(),
                "selected_depth": result[2].tolist(),
                "selected_goal": result[4]["selected_grasp_goal_world_m"].tolist(),
                "selected_sample_id": result[4]["selected_sample_id"].tolist(),
            }
        )
        values.append(result)
        (tmp_path / "measured_station_proposal.json").write_text(
            json.dumps(
                {
                    "station_poses": result[0].tolist(),
                    "selected_grasp_orientation": result[1].tolist(),
                    "pocket_depth_preference": result[2].tolist(),
                },
                indent=2,
            )
        )
        (tmp_path / "recorded_measured_joint.json").write_text(json.dumps(report, indent=2))
        print(report[-1], flush=True)
    if len(values) == 2:
        torch.testing.assert_close(values[1][3], values[0][3], atol=0, rtol=0)
        for actual, reference in zip(values[1][:3], values[0][:3], strict=True):
            torch.testing.assert_close(actual, reference, atol=1e-6, rtol=0, equal_nan=True)
