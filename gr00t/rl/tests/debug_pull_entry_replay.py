"""Offline failed-entry reconstruction and CPU/GPU halfspace audit; no simulator."""

import argparse
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import torch

from gr00t.rl.auto_open import pull_entry_gpu as gpu


def main():
    from pxr import Usd
    from gr00t.rl.auto_open.geometry import ROBOT_COLLIDER_SPECS, _read_collider_points

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1700)
    parser.add_argument("--max-route-nodes", type=int, default=None)
    parser.add_argument("--full-plan", action="store_true")
    parser.add_argument("--grid-audit", action="store_true")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "datasets/auto_open/canonical/auto_open_v1_seeds1700-1703_canonical_no_camera_20261022T102625"
        ),
    )
    args = parser.parse_args()
    rank = args.seed - 1700
    metadata = json.loads(
        (args.root / f"logs/rank{rank}-control/seed_{args.seed}/metadata.json").read_text()
    )
    shard = args.root / f"shards/episode-seed{args.seed}"
    table = pq.read_table(shard / "quarantine/raw_audit/chunk-000/episode_000000.parquet")
    row = table.slice(table.num_rows - 1).to_pylist()[0]
    provenance = json.loads((shard / "meta/simulation_provenance.json").read_text())
    episode = next(iter(provenance["episodes"].values()))
    prims = episode["door_asset"]["prims"]
    origin = np.asarray(metadata["robot_state"]["initial_base_pose_w"][:3]) - np.asarray(
        metadata["robot_state"]["initial_base_pose_env"][:3]
    )
    hinge_attributes = next(p for p in prims if p["path"].endswith("/hinge_joint"))["attributes"]
    hinge = np.asarray(hinge_attributes["physics:localPos0"]) + origin
    hinge_quat = hinge_attributes.get(
        "physics:localRot0", dict(real=1.0, imaginary=[0.0, 0.0, 0.0])
    )
    hinge_sign = float(
        np.sign(
            Rotation.from_quat(hinge_quat["imaginary"] + [hinge_quat["real"]]).apply(
                [0.0, 0.0, 1.0]
            )[2]
        )
    )
    panel = next(p for p in prims if p["path"].endswith("/door_panel/panel"))["attributes"]
    history = metadata["acceptance_diagnostics"]["pull_panel_result"]["cycles"][0]
    station = metadata["acceptance_diagnostics"]["pull_clear_sweep_plan"]
    sample = np.asarray(history["candidate_samples"], dtype=np.float32)
    width = np.float32(history["tip_radius_m"] + float(sample[0, 1]))
    tangent = np.asarray(panel["xformOp:translate"]) + origin - hinge
    tangent[[0, 2]] = 0.0
    tangent /= np.linalg.norm(tangent)
    controller = SimpleNamespace(
        fixed_finger_body_idx=0,
        moving_finger_body_idx=1,
        hinge_axis_sign=hinge_sign,
        hinge_idx=0,
        door_normal=np.array([1.0, 0.0, 0.0]),
        panel_hinge_to_free_direction=tangent,
        hinge_world=np.asarray(hinge),
        panel_hinge_to_free_edge=float(width),
        panel_center_world=np.asarray(panel["xformOp:translate"]) + origin,
        panel_half_thickness=panel["xformOp:scale"][0],
    )
    stage = Usd.Stage.Open("gr00t/rl/data/robots/b2z1_lab30/b2z1.usd")
    controller.center_geometry = {}
    for body_name, key, tokens in ROBOT_COLLIDER_SPECS:
        if key == "tcp":
            continue
        body = next(
            p
            for p in Usd.PrimRange(stage.GetDefaultPrim(), Usd.TraverseInstanceProxies())
            if p.GetName().lower() == body_name.lower()
        )
        controller.center_geometry[key] = _read_collider_points(stage, body, tokens)
    with contextlib.redirect_stdout(io.StringIO()):
        cache = gpu.PullEntryGPU.from_controllers(
            [controller], device="cuda", max_samples=1024, max_route_nodes=args.max_route_nodes
        )
    tool = np.asarray(row["ee_pose_world_xyzw"])
    ee_rotation = (
        Rotation.from_quat(tool[3:]).as_matrix() @ Rotation.from_euler("x", -np.pi / 2).as_matrix()
    )
    fixed_rotation = ee_rotation @ Rotation.from_euler("x", np.pi / 2).as_matrix()
    moving_rotation = ee_rotation @ Rotation.from_euler("y", row["gripper_feedback"]).as_matrix()

    def pose(position, rotation):
        q = Rotation.from_matrix(rotation).as_quat()
        return np.r_[position, q[[3, 0, 1, 2]], np.zeros(6)]

    bodies = np.stack(
        (
            pose(tool[:3] - fixed_rotation @ np.array([0.099999994, 0.0, 0.0]), fixed_rotation),
            pose(tool[:3], moving_rotation),
        )
    )
    root = np.r_[
        row["root_position_world"],
        np.asarray(row["root_quaternion_xyzw"])[[3, 0, 1, 2]],
        np.zeros(6),
    ]

    def tensor(value):
        return torch.tensor(np.asarray(value), device="cuda", dtype=torch.float32)

    ee_q = Rotation.from_matrix(ee_rotation).as_quat()[[3, 0, 1, 2]]
    state = dict(
        ee_pos=tensor(tool[None, :3]),
        ee_orn=tensor(ee_q[None]),
        root_states=tensor(root[None]),
        robot_body_state_w=tensor(bodies[None]),
        door_joint_position=tensor([[row["door_hinge_position"]]]),
    )
    capture = {}
    original_route = gpu._route

    def intercept(start, end, offsets, rectangles, **kwargs):
        capture.update(start=start, end=end, offsets=offsets, rectangles=rectangles, kwargs=kwargs)
        if kwargs.get("diagnostics") is not None:
            kwargs["diagnostics"].update(
                {
                    key: torch.zeros(len(start), device=start.device, dtype=torch.bool)
                    for key in ("start_free", "end_free", "direct")
                }
            )
        route = end[:, None].expand(-1, kwargs["max_nodes"], -1).clone()
        route[:, 0] = start
        return (
            route,
            torch.full((len(start),), 2, device=start.device, dtype=torch.long),
            torch.zeros(len(start), device=start.device, dtype=torch.bool),
        )

    gpu._route = intercept
    try:
        result = cache.plan(
            state,
            torch.tensor([0], device="cuda"),
            {
                key: tensor(sample[:, index][None])
                for index, key in enumerate(("height_m", "insertion_m", "yaw_rad"))
            },
            frame_edge_m=tensor([station["frame_free_boundary_from_hinge_m"]]),
            wall_front_m=tensor([station["wall_front_from_hinge_m"]]),
            hardware_height_bounds=tensor([history["hardware_height_bounds_m"]]),
        )
    finally:
        gpu._route = original_route
    start, end = capture["start"].double(), capture["end"].double()
    endpoints = torch.stack((start, end), 1)
    print(
        "RECONSTRUCTED",
        dict(
            seed=args.seed,
            ee=state["ee_pos"].tolist(),
            ee_q=state["ee_orn"].tolist(),
            hinge=hinge.tolist(),
            width=float(width),
            geometry_safe=result["geometry_safe"].tolist(),
        ),
        flush=True,
    )
    cpu_free = np.ones((len(start), 2), dtype=bool)
    gpu_free = torch.ones((len(start), 2), device="cuda", dtype=torch.bool)
    mismatch_count = 0
    grid_obstacles = []
    for index, (offset, topology) in enumerate(zip(capture["offsets"], cache.parts)):
        first, _, mask = gpu._silhouette(offset.double(), topology)
        count = mask.sum(1)
        # Pack all contour vertices without truncation; false slots repeat one.
        chosen = mask.to(torch.int32).argsort(1, descending=True)
        p = first.gather(1, chosen[..., None].expand(-1, -1, 3))[..., :2]
        valid = mask.gather(1, chosen)
        p = torch.where(valid[..., None], p, p[:, :1])
        polygon = gpu._ordered(p)
        for wall_index, rectangle in enumerate(capture["rectangles"]):
            obstacle = gpu._halfspaces(polygon, gpu._ordered(rectangle.double()))
            available = gpu._free(endpoints, [obstacle])
            gpu_free &= available
            expected = []
            for candidate in range(len(start)):
                vertices = (
                    (
                        rectangle[candidate].double()[:, None, :]
                        - offset[candidate].double()[None, :, :2]
                    )
                    .flatten(0, 1)
                    .cpu()
                    .numpy()
                )
                eq = ConvexHull(vertices).equations
                if candidate == 0:
                    grid_obstacles.append((eq, vertices.min(0), vertices.max(0)))
                residual = endpoints[candidate].cpu().numpy() @ eq[:, :2].T + eq[:, 2]
                expected.append(~(residual < -1e-8).all(1))
            expected = np.asarray(expected)
            cpu_free &= expected
            actual = available.cpu().numpy()
            if not np.array_equal(actual, expected) or not expected.all():
                print(
                    "OBSTACLE",
                    cache.collider_metadata[index]["name"],
                    wall_index,
                    "gpu",
                    actual.tolist(),
                    "scipy",
                    expected.tolist(),
                    "contour_counts",
                    count.tolist(),
                    flush=True,
                )
            mismatch_count += int(np.count_nonzero(actual != expected))
    print(
        "ENDPOINT_FLAGS",
        dict(gpu=gpu_free.tolist(), scipy=cpu_free.tolist(), mismatches=mismatch_count),
        flush=True,
    )
    if args.grid_audit:
        from scipy.ndimage import label

        start_xy, end_xy = endpoints[0].cpu().numpy()
        low, high = np.minimum(start_xy, end_xy) - 0.35, np.maximum(start_xy, end_xy) + 0.35
        x, y = np.arange(low[0], high[0] + 0.005, 0.005), np.arange(low[1], high[1] + 0.005, 0.005)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        points = np.column_stack((xx.ravel(), yy.ravel()))
        free = np.ones(len(points), dtype=bool)
        for eq, lo, hi in grid_obstacles:
            selected = np.flatnonzero(free & (points > lo).all(1) & (points < hi).all(1))
            free[selected] &= ~((points[selected] @ eq[:, :2].T + eq[:, 2]) < -1e-8).all(1)
        labels, _ = label(free.reshape(xx.shape))
        start_index = np.unravel_index(np.linalg.norm(points - start_xy, axis=1).argmin(), xx.shape)
        end_index = np.unravel_index(np.linalg.norm(points - end_xy, axis=1).argmin(), xx.shape)
        print(
            "CPU_GRID",
            dict(
                seed=args.seed,
                shape=xx.shape,
                free=int(free.sum()),
                start_label=int(labels[start_index]),
                end_label=int(labels[end_index]),
                connected=bool(labels[start_index] and labels[start_index] == labels[end_index]),
            ),
            flush=True,
        )
    if args.full_plan:
        original_solve = gpu._solve_visibility

        def report_solve(points, available, obstacles, max_nodes, query_chunk):
            route, length, safe = original_solve(
                points, available, obstacles, max_nodes, query_chunk
            )
            print(
                "GRAPH",
                dict(
                    nodes=points.shape[1],
                    available=available.sum(1).tolist(),
                    path_nodes=length.tolist(),
                    safe=safe.tolist(),
                ),
                flush=True,
            )
            return route, length, safe

        gpu._solve_visibility = report_solve
        attempt = next(i for i, ok in enumerate(result["geometry_safe"][0].tolist()) if ok)
        full = cache.plan(
            state,
            torch.tensor([0], device="cuda"),
            {
                key: tensor(sample[attempt : attempt + 1, index][None])
                for index, key in enumerate(("height_m", "insertion_m", "yaw_rad"))
            },
            frame_edge_m=tensor([station["frame_free_boundary_from_hinge_m"]]),
            wall_front_m=tensor([station["wall_front_from_hinge_m"]]),
            hardware_height_bounds=tensor([history["hardware_height_bounds_m"]]),
        )
        print(
            "FULL_PLAN",
            dict(
                seed=args.seed,
                attempt=attempt,
                **{
                    key: full[key].tolist()
                    for key in (
                        "feasible",
                        "route_safe",
                        "collision_safe",
                        "route_start_free",
                        "route_end_free",
                        "route_direct",
                        "lengths",
                    )
                },
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
