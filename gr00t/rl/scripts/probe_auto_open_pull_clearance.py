"""Offline collision-envelope clearance for previously probed pull stations."""

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

from gr00t.rl.auto_open.pull_station import collider_local_corners, PullStationConfig


def rotation(pose):
    return Rotation.from_quat(np.asarray(pose)[[4, 5, 6, 3]])


def recorded_collision_clouds(stage, poses):
    """Collision envelopes with articulation order verified by joint anchors."""
    children = defaultdict(list)
    joints = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            a = prim.GetRelationship('physics:body0').GetTargets()
            b = prim.GetRelationship('physics:body1').GetTargets()
            if a and b:
                children[str(a[0])].append(str(b[0]))
                joints.append((prim, str(a[0]), str(b[0])))
    base = next(p for p in stage.Traverse() if p.GetName() == 'base_link' and p.HasAPI(UsdPhysics.RigidBodyAPI))
    queue = deque([str(base.GetPath())])
    order = []
    while queue:
        name = queue.popleft()
        order.append(name)
        queue.extend(children[name])
    if len(order) != len(poses):
        raise ValueError('Recorded body count differs from articulation graph')
    indices = {name: i for i, name in enumerate(order)}
    anchor_errors = []
    for prim, a, b in joints:
        anchors = []
        for side, body in enumerate((a, b)):
            local = prim.GetAttribute(f'physics:localPos{side}').Get()
            pose = poses[indices[body]]
            anchors.append(rotation(pose).apply(np.asarray(local)) + pose[:3])
        anchor_errors.append(np.linalg.norm(anchors[0] - anchors[1]))
    if max(anchor_errors) > .01:
        raise ValueError(f'Body-order validation failed: {max(anchor_errors)} m')
    transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    clouds = []
    for path, index in indices.items():
        body = stage.GetPrimAtPath(path)
        inverse = transforms.GetLocalToWorldTransform(body).GetInverse()
        points = []
        for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI) or prim.GetAttribute('physics:collisionEnabled').Get() is False:
                continue
            transform = transforms.GetLocalToWorldTransform(prim) * inverse
            points.extend(transform.Transform(Gf.Vec3d(*p)) for p in collider_local_corners(prim).tolist())
        if points:
            points = np.asarray(points)
            low, high = points.min(0), points.max(0)
            corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
            clouds.append((body.GetName(), rotation(poses[index]).apply(corners) + poses[index, :3]))
    return clouds, max(anchor_errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    probe = json.loads(args.probe.read_text())
    dataset = Path(probe['dataset'])
    results = []
    config = PullStationConfig()
    for seed in sorted({r['seed'] for r in probe['results']}):
        metadata = json.loads(next(dataset.glob(f'logs/rank*-control/seed_{seed}/metadata.json')).read_text())
        provenance = json.loads((dataset / f'shards/episode-seed{seed}/meta/simulation_provenance.json').read_text())
        episode = next(iter(provenance['episodes'].values()))
        cycle = metadata['acceptance_diagnostics']['pull_panel_result']['cycles'][0]
        root = np.asarray(cycle['entry_root_state'])
        poses = np.asarray(cycle['entry_robot_body_poses_world_wxyz'])
        stage = Usd.Stage.Open(provenance['robot_asset']['path'])
        clouds, anchor_error = recorded_collision_clouds(stage, poses)
        prims = episode['door_asset']['prims']
        hinge_attr = next(p for p in prims if p['path'].endswith('/hinge_joint'))['attributes']
        panel = next(p for p in prims if p['path'].endswith('/door_panel/panel'))['attributes']
        origin = np.asarray(metadata['robot_state']['initial_base_pose_w'][:3]) - metadata['robot_state']['initial_base_pose_env'][:3]
        hinge = origin + hinge_attr['physics:localPos0']
        tangent = origin + panel['xformOp:translate'] - hinge
        tangent[[0, 2]] = 0.
        tangent /= np.linalg.norm(tangent)
        boundaries = metadata['acceptance_diagnostics']['pull_clear_sweep_plan']
        wall = boundaries['wall_front_from_hinge_m']
        frame = boundaries['frame_free_boundary_from_hinge_m']
        old_yaw = rotation(root).as_euler('xyz')[2]
        for station in probe['results']:
            if station['seed'] != seed or station['position_angle_deg'] != 10.:
                continue
            turn = Rotation.from_euler('z', np.deg2rad(station['yaw_deg']) - old_yaw)
            center = np.array([*station['position_xy_world'], root[2]])
            clearances = []
            for name, points in clouds:
                moved = turn.apply(points - root[:3]) + center
                relative = moved - hinge
                clearances.append(dict(body=name, wall_m=float(relative[:, 0].min() - wall),
                                       frame_m=float((relative @ tangent).min() - frame)))
            trunk = next(c for c in clearances if c['body'] == 'base_link')
            full_wall = min(clearances, key=lambda c: c['wall_m'])
            full_frame = min(clearances, key=lambda c: c['frame_m'])
            results.append(dict(seed=seed, heading_angle_deg=station['heading_angle_deg'],
                trunk_wall_mm=trunk['wall_m'] * 1000, trunk_frame_mm=trunk['frame_m'] * 1000,
                trunk_planning_pass=bool(trunk['wall_m'] >= config.wall_clearance_m + config.planning_margin_m
                                        and trunk['frame_m'] >= config.clearance_m + config.planning_margin_m),
                whole_robot_wall_mm=full_wall['wall_m'] * 1000, closest_wall_body=full_wall['body'],
                whole_robot_frame_mm=full_frame['frame_m'] * 1000, closest_frame_body=full_frame['body'],
                whole_robot_planning_pass=bool(full_wall['wall_m'] >= config.wall_clearance_m + config.planning_margin_m
                                              and full_frame['frame_m'] >= config.clearance_m + config.planning_margin_m),
                body_order_max_anchor_error_mm=anchor_error * 1000))
    output = dict(source_probe=str(args.probe), assumptions='Recorded retracted posture rigidly moved to proposed XY/yaw; collision bounding boxes; no simulation or path clearance test.',
                  required_planning_wall_mm=70, required_planning_frame_mm=50, results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
