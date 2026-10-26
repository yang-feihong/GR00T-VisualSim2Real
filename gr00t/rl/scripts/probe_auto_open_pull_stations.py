"""Offline station-grid evaluation with the production GPU inverse, no simulator."""

import argparse
import json
import math
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import torch
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

from gr00t.rl.auto_open.first_pull_entry import plan_first_pull_entry
from gr00t.rl.auto_open.panel_motion import PullPanelConfig
from gr00t.rl.auto_open.pull_entry_gpu import PullEntryGPU, quat_rotate
from gr00t.rl.auto_open.pull_station import pull_station_parameter_grid
from gr00t.rl.envs.door.b2z1_reset_geometry import b2z1_gripper_pose_and_jacobian
from legged_gym.utils.ee_steady_inverse_gpu import batched_steady_ee_inverse_no_gravity, _matrix_to_quaternion_wxyz


def load_canonical_records(args):
    """Recover station-entry posture from canonical audits, including failed plans."""
    from collections import defaultdict, deque
    from gr00t.rl.data_collection.auto_open_standard import B2_JOINT_NAMES
    from gr00t.rl.auto_open.pull_station import collider_local_corners
    records = []
    for seed in args.seeds:
        metadata = json.loads(next(args.dataset.glob(f'logs/rank*-control/seed_{seed}/metadata.json')).read_text())
        shard = args.dataset / 'shards' / f'episode-seed{seed}'
        provenance = json.loads((shard / 'meta/simulation_provenance.json').read_text())
        episode = next(iter(provenance['episodes'].values()))
        station = metadata['acceptance_diagnostics'].get('pull_clear_sweep_plan', {})
        history = station.get('candidate_planning', [])
        if not history:
            raise ValueError(f'Seed {seed} never reached initial station planning')
        entry = history[0]
        root = np.asarray(entry['entry_root_state'])
        arm = np.asarray(entry['entry_arm_joint_position_rad'])
        table = pq.read_table(next(shard.rglob('raw_audit/**/*.parquet')),
            columns=['root_position_world', 'arm_joint_position', 'b2_joint_position',
                     'gripper_feedback', 'door_hinge_position', 'ee_pose_world_xyzw'])
        rows = table.to_pylist()
        row = min(rows, key=lambda r: np.linalg.norm(np.asarray(r['root_position_world']) - root[:3])
                  + np.linalg.norm(np.asarray(r['arm_joint_position']) - arm))
        snapshot_error = (np.linalg.norm(np.asarray(row['root_position_world']) - root[:3])
                          + np.linalg.norm(np.asarray(row['arm_joint_position']) - arm))
        if snapshot_error > .01:
            raise ValueError(f'Seed {seed}: station snapshot mismatch {snapshot_error}')
        stage = Usd.Stage.Open(provenance['robot_asset']['path'])
        angles = dict(zip(B2_JOINT_NAMES, row['b2_joint_position']))
        angles.update(zip([f'joint{i}' for i in range(1, 7)], arm))
        angles['jointGripper'] = row['gripper_feedback']
        children = defaultdict(list)
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Joint):
                a = prim.GetRelationship('physics:body0').GetTargets()
                b = prim.GetRelationship('physics:body1').GetTargets()
                if a and b:
                    children[str(a[0])].append((prim, str(b[0])))
        base = next(p for p in stage.Traverse() if p.GetName() == 'base_link' and p.HasAPI(UsdPhysics.RigidBodyAPI))
        def frame(prim, side):
            value = np.eye(4)
            q = prim.GetAttribute(f'physics:localRot{side}').Get()
            value[:3, :3] = Rotation.from_quat([*q.GetImaginary(), q.GetReal()]).as_matrix()
            value[:3, 3] = prim.GetAttribute(f'physics:localPos{side}').Get()
            return value
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat(root[[4, 5, 6, 3]]).as_matrix()
        transform[:3, 3] = root[:3]
        transforms = {str(base.GetPath()): transform}
        queue, poses = deque([str(base.GetPath())]), []
        while queue:
            path = queue.popleft()
            value = transforms[path]
            q = Rotation.from_matrix(value[:3, :3]).as_quat()
            poses.append(np.r_[value[:3, 3], q[[3, 0, 1, 2]]])
            for joint, child in children[path]:
                motion = np.eye(4)
                if joint.IsA(UsdPhysics.RevoluteJoint):
                    axis = joint.GetAttribute('physics:axis').Get().lower()
                    motion[:3, :3] = Rotation.from_euler(axis, angles.get(joint.GetName(), 0.)).as_matrix()
                elif joint.IsA(UsdPhysics.PrismaticJoint):
                    motion['XYZ'.index(joint.GetAttribute('physics:axis').Get()), 3] = angles.get(joint.GetName(), 0.)
                transforms[child] = value @ frame(joint, 0) @ motion @ np.linalg.inv(frame(joint, 1))
                queue.append(child)
        prims = episode['door_asset']['prims']
        ha = next(p['attributes'] for p in prims if p['path'].endswith('/hinge_joint'))
        panel = next(p['attributes'] for p in prims if p['path'].endswith('/door_panel/panel'))
        origin = np.asarray(metadata['robot_state']['initial_base_pose_w'][:3]) - metadata['robot_state']['initial_base_pose_env'][:3]
        hinge = origin + ha['physics:localPos0']
        hq = ha.get('physics:localRot0', dict(real=1., imaginary=[0., 0., 0.]))
        sign = float(np.sign(Rotation.from_quat(hq['imaginary'] + [hq['real']]).apply([0, 0, 1])[2]))
        center = np.asarray(panel['xformOp:translate']) + origin
        tangent = center - hinge
        tangent[[0, 2]] = 0.
        tangent /= np.linalg.norm(tangent)
        snapshot = Usd.Stage.CreateInMemory()
        for spec in prims:
            prim = snapshot.DefinePrim(spec['path'], spec['type'])
            for name, value in spec['attributes'].items():
                if name == 'xformOpOrder':
                    prim.CreateAttribute(name, Sdf.ValueTypeNames.TokenArray).Set(value)
                elif name.startswith('xformOp:'):
                    if isinstance(value, dict):
                        prim.CreateAttribute(name, Sdf.ValueTypeNames.Quatd).Set(Gf.Quatd(value['real'], Gf.Vec3d(*value['imaginary'])))
                    else:
                        prim.CreateAttribute(name, Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(*value))
                elif name in ('axis', 'height', 'radius', 'size'):
                    prim.GetAttribute(name).Set(value)
        xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
        hardware = []
        for spec in prims:
            if (any(k in spec['path'] for k in ('/door_handle/', '/latch_link/'))
                    and spec['attributes'].get('physics:collisionEnabled') is True):
                prim = snapshot.GetPrimAtPath(spec['path'])
                matrix = xforms.GetLocalToWorldTransform(prim)
                hardware.extend(matrix.Transform(Gf.Vec3d(*p))[2] + origin[2] for p in collider_local_corners(prim))
        records.append(dict(seed=seed, root=root, arm=arm, hinge=hinge,
            asset_path=provenance['robot_asset']['path'], body_poses=np.asarray(poses),
            normal=[1., 0., 0.], tangent=tangent, width=episode['door_randomization']['doorWidth'],
            planner_width=float((center - hinge) @ tangent + panel['xformOp:scale'][1]),
            center=center, half=panel['xformOp:scale'][0], sign=sign, door_angle=row['door_hinge_position'],
            hardware=[min(hardware), max(hardware)],
            frame=station['frame_free_boundary_from_hinge_m'], wall=station['wall_front_from_hinge_m'],
            snapshot_error=snapshot_error))
    return records


def load_records(args):
    shard = args.dataset / 'shards' / f'episode-seed{args.seeds[0]}'
    audit = next(shard.rglob('raw_audit/**/*.parquet'))
    if 'post_root_state' not in pq.read_schema(audit).names:
        return load_canonical_records(args)
    records = []
    asset_path = None
    arm_names = [f'joint{i}' for i in range(1, 7)]
    for seed in args.seeds:
        metadata = next(args.dataset.glob(f'logs/rank*-control/seed_{seed}/metadata.json'))
        m = json.loads(metadata.read_text())
        shard = args.dataset / 'shards' / f'episode-seed{seed}'
        provenance = json.loads((shard / 'meta/simulation_provenance.json').read_text())
        episode = next(iter(provenance['episodes'].values()))
        asset_path = provenance['robot_asset']['path']
        c = m['acceptance_diagnostics']['pull_panel_result']['cycles'][0]
        root = np.asarray(c['entry_root_state'])
        rows = pq.read_table(next(shard.rglob('raw_audit/chunk-000/episode_000000.parquet')),
            columns=['autoopen_stage', 'root_position_world', 'gripper_ee_pose_world_xyzw',
                     'arm_joint_position', 'door_hinge_position', 'post_root_state',
                     'post_all_joint_position', 'post_gripper_ee_pose_world_xyzw',
                     'post_door_hinge_position']).to_pylist()
        arm_ids = [episode['robot_joint_names'].index(name) for name in arm_names]
        snapshots = []
        for row in rows:
            snapshots.append(dict(root=row['root_position_world'], ee=row['gripper_ee_pose_world_xyzw'],
                                  arm=row['arm_joint_position'], door=row['door_hinge_position']))
            if row['post_root_state'] is not None:
                snapshots.append(dict(root=row['post_root_state'][:3], ee=row['post_gripper_ee_pose_world_xyzw'],
                    arm=np.asarray(row['post_all_joint_position'])[arm_ids].tolist(), door=row['post_door_hinge_position']))
        def snapshot_error(row):
            return (np.linalg.norm(np.asarray(row['root']) - root[:3])
                    + np.linalg.norm(np.asarray(row['ee'][:3]) - c['entry_ee_position_world_m']))
        row = min(snapshots, key=snapshot_error)
        if snapshot_error(row) > 1.e-5:
            raise ValueError(f'seed {seed}: no matching recorded entry snapshot')
        prims = episode['door_asset']['prims']
        ha = next(p for p in prims if p['path'].endswith('/hinge_joint'))['attributes']
        panel = next(p for p in prims if p['path'].endswith('/door_panel/panel'))['attributes']
        origin = np.asarray(m['robot_state']['initial_base_pose_w'][:3]) - np.asarray(m['robot_state']['initial_base_pose_env'][:3])
        hinge = origin + ha['physics:localPos0']
        hq = ha.get('physics:localRot0', dict(real=1., imaginary=[0., 0., 0.]))
        sign = float(np.sign(Rotation.from_quat(hq['imaginary'] + [hq['real']]).apply([0, 0, 1])[2]))
        center = np.asarray(panel['xformOp:translate']) + origin
        tangent = center - hinge
        tangent[[0, 2]] = 0.
        tangent /= np.linalg.norm(tangent)
        width = episode['door_randomization']['doorWidth']
        # The online planner uses the free-edge station radius, not twice the
        # panel half-width (the asset has a hinge-side gap).
        candidates = c['candidate_samples']
        valid = [i for i, value in enumerate(c['max_position_error_m']) if value is not None]
        selected = c['selected_attempt']
        if selected < 0:
            selected = min(valid, key=lambda i: max(c['max_position_error_m'][i] / .004,
                                                   c['max_orientation_error_rad'][i] / math.radians(1.)))
        planner_width = c['tip_radius_m'] + candidates[selected][1]
        records.append(dict(seed=seed, root=root, arm=row['arm'], hinge=hinge, asset_path=asset_path,
            body_poses=c['entry_robot_body_poses_world_wxyz'],
            normal=[1., 0., 0.], tangent=tangent, width=width, planner_width=planner_width,
            center=center, half=panel['xformOp:scale'][0], sign=sign,
            door_angle=row['door'], hardware=c['hardware_height_bounds_m'],
            frame=m['acceptance_diagnostics']['pull_clear_sweep_plan']['frame_free_boundary_from_hinge_m'],
            wall=m['acceptance_diagnostics']['pull_clear_sweep_plan']['wall_front_from_hinge_m']))
    return records


def evaluate_stations(args, records, xy=None, yaw=None, configurations=None):
    start = time.perf_counter()
    arm_names = [f'joint{i}' for i in range(1, 7)]
    stage = Usd.Stage.Open(records[0]['asset_path'])
    joints = {p.GetName(): p for p in stage.Traverse() if p.GetName() in arm_names}
    limits = torch.tensor([[math.radians(joints[name].GetAttribute('physics:lowerLimit').Get()),
                            math.radians(joints[name].GetAttribute('physics:upperLimit').Get())]
                           for name in arm_names], dtype=torch.float32, device=args.device)

    def tensor(key):
        return torch.tensor(np.asarray([r[key] for r in records]), device=args.device, dtype=torch.float32)

    if xy is None:
        xy, yaw, configurations = pull_station_parameter_grid(tensor('hinge')[:, :2], tensor('normal')[:, :2],
            tensor('tangent')[:, :2], tensor('width'), position_angles_deg=args.position_angles_deg,
            heading_offsets_deg=args.heading_offsets_deg)
    station_count = len(configurations)
    roots = tensor('root')[:, None].expand(-1, station_count, -1).clone()
    old_rotation = Rotation.from_quat(tensor('root')[:, [4, 5, 6, 3]].cpu().numpy())
    old_yaw = old_rotation.as_euler('xyz')[:, 2]
    rotations = Rotation.from_euler('z', (yaw.cpu().numpy() - old_yaw[:, None]).flatten()) * Rotation.from_quat(
        np.repeat(old_rotation.as_quat(), station_count, axis=0))
    roots[..., :2] = xy
    roots[..., 3:7] = torch.tensor(rotations.as_quat()[:, [3, 0, 1, 2]].reshape(-1, station_count, 4),
                                  device=args.device, dtype=torch.float32)
    roots = roots.flatten(0, 1)
    arms = tensor('arm')[:, None].expand(-1, station_count, -1).flatten(0, 1)
    ee, rotation, _ = b2z1_gripper_pose_and_jacobian(roots[:, :3], roots[:, [4, 5, 6, 3]], arms)
    state = dict(ee_pos=ee, ee_orn=_matrix_to_quaternion_wxyz(rotation), root_states=roots,
                 door_joint_position=tensor('door_angle').repeat_interleave(station_count)[:, None])
    cache = SimpleNamespace(max_samples=1024)
    cache._sample = MethodType(PullEntryGPU._sample, cache)
    for target, key in (('hinge_world', 'hinge'), ('door_normal', 'normal'), ('tangent', 'tangent'),
                        ('panel_width', 'planner_width'), ('panel_center', 'center'),
                        ('panel_half_thickness', 'half'), ('hinge_axis_sign', 'sign')):
        setattr(cache, target, tensor(key).repeat_interleave(station_count, dim=0))
    cache.hinge_idx = torch.zeros(len(roots), device=args.device, dtype=torch.int64)
    config = PullPanelConfig()
    grid = torch.cartesian_prod(torch.tensor(config.height_offsets_m, device=args.device, dtype=torch.float64),
                               torch.tensor(config.insertion_lengths_m, device=args.device, dtype=torch.float64))
    distance = grid[:, 0].square() + (grid[:, 1] - .04).square()
    grid = grid[torch.argsort(distance, stable=True)]
    grid[:, 0] += config.height_nominal_m
    samples = dict(height_m=grid[:, 0].float()[None].expand(len(roots), -1),
                   insertion_m=grid[:, 1].float()[None].expand(len(roots), -1))
    geometry = plan_first_pull_entry(cache, state, torch.arange(len(roots), device=args.device), samples,
        frame_edge_m=tensor('frame').repeat_interleave(station_count), wall_front_m=tensor('wall').repeat_interleave(station_count),
        hardware_height_bounds=tensor('hardware').repeat_interleave(station_count, dim=0), config=config)
    print(f'GPU evaluation: {len(records)} seeds x {station_count} stations x 27 candidates', flush=True)

    def inverse(positions, quaternions, owner):
        result = batched_steady_ee_inverse_no_gravity(positions, quaternions,
            base_position_world=roots[owner, :3], base_orientation_world_xyzw=roots[owner][:, [4, 5, 6, 3]],
            initial_arm_joint_positions=arms[owner], joint_limits=limits)
        position_error = torch.linalg.vector_norm(result.position_error, dim=-1)
        angle_error = 2. * torch.asin(torch.linalg.vector_norm(result.orientation_error, dim=-1).clamp(0., 1.))
        passed = (position_error <= .004) & (angle_error <= math.radians(1.))
        return passed, position_error, angle_error

    station, candidate = torch.where(geometry['feasible'])
    p, q = geometry['endpoint'][station, candidate], geometry['orientation'][station, candidate]
    axis = quat_rotate(q, torch.tensor([1., 0., 0.], device=args.device))
    alignment = p - axis * (.03 + samples['insertion_m'][station, candidate])[:, None]
    point_pass, point_error, point_angle = inverse(torch.cat((alignment, p)), torch.cat((q, q)), station.repeat(2))
    n = len(station)
    lengths = geometry['lengths'][station, candidate]
    point_ok = point_pass[:n] & point_pass[n:]
    lengths = torch.where(point_ok, lengths, 0)
    path_owner, path_sample = torch.where(torch.arange(1024, device=args.device)[None] < lengths[:, None])
    print(f'Whole-path inverse batch: {len(path_owner)} poses', flush=True)
    passed, error, angle = inverse(geometry['positions'][station[path_owner], candidate[path_owner], path_sample],
                                  geometry['orientations'][station[path_owner], candidate[path_owner], path_sample],
                                  station[path_owner])
    errors = torch.zeros(n, device=args.device).scatter_reduce_(0, path_owner, error, reduce='amax')
    angles = torch.zeros(n, device=args.device).scatter_reduce_(0, path_owner, angle, reduce='amax')
    errors = errors.masked_fill(~point_ok, torch.inf)
    angles = angles.masked_fill(~point_ok, torch.inf)
    path_pass = (errors <= .004) & (angles <= math.radians(1.))
    output = dict(dataset=str(args.dataset), assumptions='Recorded base height/tilt and retracted joints; hypothetical XY/yaw. No physics simulation.',
                  position_tolerance_m=.004, orientation_tolerance_deg=1., results=[])
    for k, record in enumerate(records):
        for j, pair in enumerate(configurations.cpu().tolist()):
            indices = torch.where(station == k * station_count + j)[0]
            good = indices[path_pass[indices]]
            pick = int(good[0]) if len(good) else None
            best = int(indices[errors[indices].argmin()]) if len(indices) and torch.isfinite(errors[indices]).any() else None
            diagnostic = None
            if best is not None:
                locations = torch.where(path_owner == best)[0]
                worst = int(locations[error[locations].argmax()])
                sample_index = int(path_sample[worst])
                diagnostic = dict(candidate=int(candidate[best]), sample_index=sample_index,
                    path_length=int(lengths[best]),
                    position_world=geometry['positions'][station[best], candidate[best], sample_index].cpu().tolist(),
                    alignment_world=alignment[best].cpu().tolist(), endpoint_world=p[best].cpu().tolist(),
                    alignment_pass=bool(point_pass[best]), endpoint_pass=bool(point_pass[n + best]),
                    position_error_mm=float(error[worst]) * 1000,
                    orientation_error_deg=math.degrees(float(angle[worst])))
            row = dict(seed=record['seed'], position_angle_deg=pair[0], heading_angle_deg=sum(pair),
                position_xy_world=xy[k, j].cpu().tolist(), yaw_deg=math.degrees(float(yaw[k, j])),
                geometry_candidates=len(indices), alignment_pass=int(point_pass[indices].sum()),
                endpoint_pass=int(point_pass[n + indices].sum()), whole_path_pass=len(good),
                min_max_position_error_mm=float(errors[indices].min()) * 1000
                    if len(indices) and torch.isfinite(errors[indices]).any() else None,
                selected_height_m=float(samples['height_m'][station[pick], candidate[pick]]) if pick is not None else None,
                selected_insertion_m=float(samples['insertion_m'][station[pick], candidate[pick]]) if pick is not None else None,
                best_candidate_diagnostic=diagnostic)
            output['results'].append(row)
            if not getattr(args, 'quiet', False):
                print(row, flush=True)
    torch.cuda.synchronize()
    output['wall_time_s'] = time.perf_counter() - start
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1700, 1701, 1702, 1703])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--position-angles-deg', type=float, nargs='+', default=[15., 20., 25.])
    parser.add_argument('--heading-offsets-deg', type=float, nargs='+', default=[0., 5., 10.])
    args = parser.parse_args()
    output = evaluate_stations(args, load_records(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(f'Report: {args.output}; {output["wall_time_s"]:.2f} s', flush=True)


if __name__ == '__main__':
    main()
