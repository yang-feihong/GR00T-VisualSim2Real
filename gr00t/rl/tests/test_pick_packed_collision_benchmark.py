"""Recorded native-mesh A/B benchmark for packed versus dense masked queries."""

import json
import math
import os
import time

import pytest
import torch
import warp as wp
from gr00t.rl.tests.test_pick_surface_worklist import full_worklist

from gr00t.rl.tests.test_pick_basket_planning_throughput import _basket_inputs


@pytest.mark.skipif(not os.environ.get("PICK_BASKET_TIMING_ROOT"), reason="Records required")
def test_packed_queries_preserve_hundred_recorded_paths(monkeypatch, tmp_path):
    from gr00t.rl.pick import (
        basket_grasp,
        planning,
        surface_worklist,
    )
    from gr00t.rl.pick import (
        collision as collider,
    )

    state, sizes, support, mask, geometry, arm, limits, minimum = _basket_inputs()
    native_environment = planning.grasp_environment_clearance

    def dense_environment(
        collision, geometry, p, q, aperture, *, query_mask=None, minimum_clearance_m=0.002
    ):
        if query_mask is None or not getattr(geometry, "supports_query_mask", False):
            return native_environment(
                collision,
                geometry,
                p,
                q,
                aperture,
                query_mask=query_mask,
                minimum_clearance_m=minimum_clearance_m,
            )
        cap = minimum_clearance_m + 0.0001
        opened = collision.gripper_clearance(
            geometry, p, q, query_mask=query_mask, maximum_distance_m=cap
        )
        closing = collision.gripper_closing_clearance(
            geometry, p, q, aperture, query_mask=query_mask, maximum_distance_m=cap
        )
        return torch.minimum(opened, closing).masked_fill(~query_mask, torch.nan)

    def dense_arm(
        self,
        geometry,
        roots,
        joints,
        *,
        maximum_distance_m=2.0,
        include_gripper=True,
        query_mask=None,
    ):
        shape = roots.shape[:-1]
        transforms = self.link_transforms(roots.reshape(-1, 13), joints.reshape(-1, 6))
        result = roots.new_full(shape, maximum_distance_m)
        for index, points in self.parts:
            if not include_gripper and index == 6:
                continue
            position, rotation = transforms[index]
            gap = geometry.rigid_surface_clearance(
                position.reshape(*shape, 3),
                collider._matrix_to_quaternion_wxyz(rotation).reshape(*shape, 4),
                points,
                check_rays=False,
                maximum_distance_m=maximum_distance_m,
                query_mask=query_mask,
            )
            result = torch.minimum(result, gap)
        return result

    work = {}
    native_launch = wp.launch

    def launch(kernel, dim, *args, **kwargs):
        if kernel.__name__ in (
            "_batched_surface",
            "_rigid_surface_clearance_kernel",
            "batched_surface",
        ):
            threads = math.prod(dim)
            work["surface_threads"] += threads
            work["peak_surface_threads"] = max(work["peak_surface_threads"], threads)
            inputs = kwargs.get("inputs", args[0] if args else None)
            work["maximum_source_vertices"] = max(work["maximum_source_vertices"], inputs[-2])
        if kernel is surface_worklist._overlapping_blocks:
            work["broadphase_threads"] += math.prod(dim)
        return native_launch(kernel, dim, *args, **kwargs)

    monkeypatch.setattr(wp, "launch", launch)

    def run():
        work.update(
            surface_threads=0,
            peak_surface_threads=0,
            maximum_source_vertices=0,
            broadphase_threads=0,
        )
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = basket_grasp.plan_basket_grasp_batch(
            state,
            sizes,
            limits,
            arm,
            geometry,
            0.02,
            support_points=support,
            support_mask=mask,
            initial_height=state["object_state"][:, 2],
            minimum_height=minimum,
            prefer_realized_body=torch.zeros(len(sizes), device=sizes.device, dtype=torch.bool),
            realized_body=torch.ones(len(sizes), device=sizes.device, dtype=torch.bool),
            minimum_realized_grasp_margin_rad=math.radians(2),
        )
        torch.cuda.synchronize()
        return result, {"planning_s": time.perf_counter() - started, **work}

    with torch.inference_mode():
        with monkeypatch.context() as scope:
            scope.setattr(planning, "grasp_environment_clearance", dense_environment)
            scope.setattr(collider.ArmCollisionSamples, "clearance", dense_arm)
            scope.setattr(surface_worklist, "surface_worklist", full_worklist)
            baseline, dense_work = run()
        packed, packed_work = run()
    torch.testing.assert_close(baseline[1], packed[1], rtol=0, atol=0)
    assert baseline[1].all()
    for before, after in zip(baseline[0], packed[0], strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    torch.testing.assert_close(baseline[2]["selected"], packed[2]["selected"], rtol=0, atol=0)
    report = {
        "physics_simulated": False,
        "cloned_recorded_rows": len(sizes),
        "valid": int(packed[1].sum()),
        "dense_masked": dense_work,
        "packed": packed_work,
        "surface_threads_definition": "Actual launched point-block threads, including tail padding",
        "paths_bitwise_equal": True,
    }
    (tmp_path / "packed_collision_comparison.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)
    assert packed_work["surface_threads"] < dense_work["surface_threads"]
    assert packed_work["planning_s"] < 60
