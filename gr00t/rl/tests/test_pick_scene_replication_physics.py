"""Opt-in original-room replication probe, including furniture surface queries."""

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.skipif(not os.environ.get("PICK_REPLICATION_PROBE"), reason="Real SDK probe")
def test_original_room_static_replication_queries(tmp_path):
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True, device="cuda:0", enable_cameras=False)
    import isaaclab.sim as sim_utils
    import omni.physx
    from gr00t.rl.pick.scene_replication import register_static_room_replication
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from gr00t.rl.pick.scene_selection import disable_embedded_physics_scenes

    records = Path(os.environ["PICK_REPLICATION_PROBE"])
    assets = list(
        dict.fromkeys(
            json.loads(path.read_text())["scene_usd_path"]
            for path in sorted(records.glob("logs/rank0-control/seed_*/episode.json"))
        )
    )
    copies = 20
    enabled = os.environ.get("PICK_REPLICATION_ENABLED", "1") != "0"
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            device="cuda:0", create_stage_in_memory=True, enable_scene_query_support=True
        )
    )
    replication = None
    report_path = tmp_path / "static_replication_queries.json"
    report = {
        "assets": assets,
        "environments": len(assets) * copies,
        "validated": False,
        "replication_enabled": enabled,
    }
    report_path.write_text(json.dumps(report, indent=2))
    try:
        with sim_utils.use_stage(sim.get_initial_stage()):
            stage = sim.get_initial_stage()
            started = time.perf_counter()
            roots, translations = [], {}
            for group, asset in enumerate(assets):
                for index in range(copies):
                    path = f"/World/Group{group}/Room{index}"
                    offset = Gf.Vec3d((index % 5) * 40, (group * 4 + index // 5) * 40, 0)
                    room = UsdGeom.Xform.Define(stage, path)
                    room.AddTranslateOp().Set(offset)
                    room.GetPrim().GetReferences().AddReference(asset)
                    roots.append((path, asset))
                    translations[path] = offset
            disable_embedded_physics_scenes(stage, roots)
            if enabled:
                replication = register_static_room_replication(stage, roots)
                assert replication is not None and len(replication.groups) == len(assets)
            cube = UsdGeom.Cube.Define(stage, "/World/QueryControl")
            cube.CreateSizeAttr(0.2)
            cube.AddTranslateOp().Set(Gf.Vec3d(-1000, 0, 0))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            report["assembly_s"] = time.perf_counter() - started
            report_path.write_text(json.dumps(report, indent=2))
            print({"phase": "assembly", "seconds": report["assembly_s"]}, flush=True)
            started = time.perf_counter()
            sim_utils.attach_stage_to_usd_context()
            report["attach_s"] = time.perf_counter() - started
            report_path.write_text(json.dumps(report, indent=2))
            print({"phase": "attach", "seconds": report["attach_s"]}, flush=True)
            sim.reset()
            sim.step(render=False)
        query = omni.physx.get_physx_scene_query_interface()
        assert query.raycast_closest((-1000, 0, 1), (0, 0, -1), 2).get("hit")
        boxes = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=False,
            ignoreVisibility=True,
        )
        probes, mismatches, queries = [], [], []
        query_count = 0
        for group in range(len(assets)):
            source = f"/World/Group{group}/Room0"
            merge_roots, requests = [], []
            for prim in Usd.PrimRange(stage.GetPrimAtPath(source), Usd.TraverseInstanceProxies()):
                if "PhysxMeshMergeCollisionAPI" in prim.GetAppliedSchemas():
                    merge_roots.append(prim)
                if (
                    len(requests) < 4
                    and prim.IsA(UsdGeom.Mesh)
                    and "/floor/" in str(prim.GetPath())
                ):
                    mesh = UsdGeom.Mesh(prim)
                    points, indices = (
                        mesh.GetPointsAttr().Get(),
                        mesh.GetFaceVertexIndicesAttr().Get(),
                    )
                    if points is None or indices is None or len(indices) < 3:
                        continue
                    center = sum((Gf.Vec3d(points[int(i)]) for i in indices[:3]), Gf.Vec3d()) / 3
                    center = (
                        UsdGeom.Xformable(prim)
                        .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                        .Transform(center)
                    )
                    requests.append((center + Gf.Vec3d(0, 0, 0.1), Gf.Vec3d(0, 0, -1), 0.2))
            assert merge_roots
            for index in np.linspace(0, len(merge_roots) - 1, min(8, len(merge_roots)), dtype=int):
                bounds = boxes.ComputeWorldBound(merge_roots[index]).ComputeAlignedRange()
                if bounds.IsEmpty():
                    continue
                center, extent = bounds.GetMidpoint(), bounds.GetSize()
                for axis in range(3):
                    for sign in (-1, 1):
                        direction = Gf.Vec3d()
                        direction[axis] = -sign
                        requests.append(
                            (
                                center - direction * (extent[axis] / 2 + 0.1),
                                direction,
                                extent[axis] + 0.2,
                            )
                        )
            hits = []
            for index in range(copies):
                path = f"/World/Group{group}/Room{index}"
                shift = translations[path] - translations[source]
                row = []
                for origin, direction, distance in requests:
                    hit = query.raycast_closest(tuple(origin + shift), tuple(direction), distance)
                    row.append(
                        {
                            "hit": bool(hit.get("hit")),
                            "collision": str(hit.get("collision", "")).replace(path, ""),
                            "position": (np.asarray(hit["position"]) - np.asarray(shift)).tolist()
                            if hit.get("hit")
                            else None,
                        }
                    )
                    query_count += 1
                hits.append(row)
            queries.append(hits)
            assert any(hit["hit"] for hit in hits[0])
            for index, row in enumerate(hits[1:], start=1):
                for request, (original, copied) in enumerate(zip(hits[0], row, strict=True)):
                    same = (
                        original["hit"] == copied["hit"]
                        and original["collision"] == copied["collision"]
                    )
                    if same and original["hit"]:
                        same = np.allclose(
                            original["position"], copied["position"], atol=2e-4, rtol=0
                        )
                    if not same:
                        mismatches.append(
                            {
                                "group": group,
                                "copy": index,
                                "request": request,
                                "original": original,
                                "copy_result": copied,
                            }
                        )
            probes.append(
                {
                    "asset": assets[group],
                    "requests": len(requests),
                    "original_hits": sum(hit["hit"] for hit in hits[0]),
                    "merge_colliders": len(merge_roots),
                }
            )
        report.update(
            query_count=query_count, probes=probes, mismatches=mismatches, queries=queries
        )
        report_path.write_text(json.dumps(report, indent=2))
        assert query_count <= 10000
        reference_path = os.environ.get("PICK_REPLICATION_REFERENCE")
        if reference_path:
            reference = json.loads(Path(reference_path).read_text())
            differences = []
            for group, (old_group, new_group) in enumerate(
                zip(reference["queries"], queries, strict=True)
            ):
                for copy, (old_row, new_row) in enumerate(zip(old_group, new_group, strict=True)):
                    for request, (old, new) in enumerate(zip(old_row, new_row, strict=True)):
                        same = old["hit"] == new["hit"] and old["collision"] == new["collision"]
                        if same and old["hit"]:
                            same = np.allclose(old["position"], new["position"], atol=2e-4, rtol=0)
                        if not same:
                            differences.append(
                                {
                                    "group": group,
                                    "copy": copy,
                                    "request": request,
                                    "reference": old,
                                    "new": new,
                                }
                            )
            report["baseline_differences"] = differences
            report_path.write_text(json.dumps(report, indent=2))
            assert not differences
        elif enabled:
            assert not mismatches
        report["validated"] = True
        report_path.write_text(json.dumps(report, indent=2))
        print(
            {
                key: value
                for key, value in report.items()
                if key not in ("queries", "mismatches", "baseline_differences")
            },
            flush=True,
        )
    finally:
        if replication is not None:
            replication.close()
        sim.clear_all_callbacks()
        sim.clear_instance()
        launcher.app.close()
