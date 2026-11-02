"""Recorded-input gaze comparison: skip only candidates already rejected by inverse."""

import ast
import inspect
import json
import os
import time
from pathlib import Path

import pytest
import torch

from gr00t.rl.tests.test_pick_recorded_batch_invariance import _recorded_batch


@pytest.mark.skipif(not os.environ.get("PICK_GAZE_TIMING_ROOT"), reason="Records required")
def test_recorded_gaze_collision_mask(tmp_path, monkeypatch):
    from gr00t.rl.auto_open.base_trajectory import BasePoseTrajectory
    from gr00t.rl.auto_open.motion_math import _quat_multiply
    from gr00t.rl.pick import gaze

    state, destination, _, _, _, geometry, collision, limits = _recorded_batch(
        Path(os.environ["PICK_GAZE_TIMING_ROOT"]), count=4
    )
    rows = torch.arange(100, device="cuda").remainder(4)
    root = state["root_states"][rows]
    w, x, y, z = root[:, 3:7].unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    base = BasePoseTrajectory(len(rows), "cuda")
    owners = torch.arange(len(rows), device="cuda")
    base.install(owners, torch.cat((root[:, :2], yaw[:, None]), -1), destination[rows, None], 0.02)
    count = int(base.length.max())
    roots = root[:, None].expand(-1, count, -1).clone()
    roots[..., :2] = base.poses[:, :count, :2]
    delta = base.poses[:, :count, 2] - yaw[:, None]
    q = torch.stack(((delta / 2).cos(), delta * 0, delta * 0, (delta / 2).sin()), -1)
    roots[..., 3:7] = _quat_multiply(q, roots[..., 3:7])
    current = gaze._gaze_inverse_samples

    class FullCollisionReference(ast.NodeTransformer):
        def visit_keyword(self, node):
            if node.arg == "query_mask":
                node.value = ast.Constant(None)
            return self.generic_visit(node)

        def visit_Assign(self, node):
            if (
                any(isinstance(t, ast.Name) and t.id == "clearance" for t in node.targets)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "where"
            ):
                node.value = ast.Name(id="clearance", ctx=ast.Load())
            return self.generic_visit(node)

    tree = FullCollisionReference().visit(ast.parse(inspect.getsource(current)))
    scope = dict(current.__globals__)
    exec(compile(ast.fix_missing_locations(tree), "full_gaze_collision_reference", "exec"), scope)
    results, report = [], []
    for name, function in (("full", scope["_gaze_inverse_samples"]), ("masked", current)):
        monkeypatch.setattr(gaze, "_gaze_inverse_samples", function)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            result = gaze.plan_approach_gaze_batch(
                roots,
                state["dof_pos"][rows][:, collision.arm_joint_ids],
                state["object_state"][rows, :3],
                limits,
                0.02,
                collision,
                geometry.select(rows),
                roots.new_zeros(len(rows), 3),
                lengths=base.length,
            )
        torch.cuda.synchronize()
        report.append(
            {
                "mode": name,
                "elapsed_s": time.perf_counter() - started,
                "valid": int(result[2]["valid"].sum()),
                "rows": len(rows),
            }
        )
        results.append(result)
        (tmp_path / "gaze_throughput.json").write_text(json.dumps(report, indent=2))
        print(report[-1], flush=True)
    for actual, expected in zip(results[1][:2], results[0][:2], strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(results[1][2]["candidate_valid"], results[0][2]["candidate_valid"])
