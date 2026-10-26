"""Offline position-only panel search; never starts a simulation or edits control."""

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
import torch


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--step', type=float, default=.05)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--refine-report', type=Path,
                        help='Reuse reconstructed poses; batch 1cm axis searches and local 1.25cm refinement.')
    parser.add_argument('--compact', action='store_true', help='Test 3 normal x 3 height offsets only.')
    args = parser.parse_args()
    start = time.perf_counter()
    failures = []
    for path in sorted(args.dataset.glob('logs/rank*-control/summary.json')):
        failures.extend(f for f in json.loads(path.read_text())['failures']
                        if f['reason'] == 'panel_contact_inverse_mapping_failed')
    if args.limit:
        failures = failures[:args.limit]
    first_shard = args.dataset / 'shards' / f"episode-seed{failures[0]['seed']}"
    provenance = json.loads((first_shard / 'meta/simulation_provenance.json').read_text())
    match = re.search(r'x0p(\d+)_y0_z0p(\d+)', provenance['robot_asset']['path'])
    if match is None:
        raise ValueError('Cannot identify recorded mount')
    recorded_mount = (float('0.' + match[1]), 0., float('0.' + match[2]))
    # Restore recorded geometry only in this isolated process, before FK/kernel imports.
    import gr00t.rl.utils.b2z1_asset as asset
    asset.ARM_MOUNT_XYZ = recorded_mount
    from pxr import Usd
    from gr00t.rl.auto_open.geometry import ROBOT_COLLIDER_SPECS, _read_collider_points
    from gr00t.rl.auto_open.tensor_stage_planning import align_vectors, _quat_rotate
    from gr00t.rl.auto_open.planning_ik import solve_candidate_ik
    from gr00t.rl.envs.door.b2z1_reset_geometry import (
        b2z1_gripper_pose_and_jacobian, quaternion_xyzw_to_matrix,
    )
    from legged_gym.utils.ee_steady_inverse_gpu import batched_steady_ee_inverse_no_gravity

    def tensor(value):
        return torch.as_tensor(np.asarray(value), dtype=torch.float64, device=args.device)

    if args.refine_report:
        report = json.loads(args.refine_report.read_text())
        if tuple(report['recorded_mount']) != recorded_mount:
            raise ValueError('Recorded mount mismatch')
        records = report['records']
        if 'joint_limits' not in report:
            usd = Usd.Stage.Open(provenance['robot_asset']['path'])
            joints = {p.GetName(): p for p in usd.Traverse() if p.GetName().startswith('joint')}
            report['joint_limits'] = [[math.radians(joints[f'joint{i}'].GetAttribute(f'physics:{name}Limit').Get())
                                       for name in ('lower', 'upper')] for i in range(1, 7)]
        limits = tensor(report['joint_limits'])
        axis = np.arange(-20, 21) * .01
        bank, owners = [], []
        for index, record in enumerate(records):
            if args.compact:
                offsets = np.asarray([(n, 0., z) for n in (0., -.05, -.10) for z in (0., .05, .10)])
                bank.extend(offsets)
                owners.extend([index] * len(offsets))
                continue
            offsets = [np.zeros(3)]
            for direction in range(3):
                for distance in axis:
                    value = np.zeros(3)
                    value[direction] = distance
                    offsets.append(value)
            selected = record['groups']['all']['offset_m']
            if selected is not None:
                offsets.extend(np.asarray(selected) + np.array([a, b, c]) * .0125
                               for a in range(-2, 3) for b in range(-2, 3) for c in range(-2, 3))
            offsets = np.unique(np.round(np.asarray(offsets), 8), axis=0)
            bank.extend(offsets)
            owners.extend([index] * len(offsets))
        owner = torch.tensor(owners, dtype=torch.int64, device=args.device)
        offset = tensor(bank)
        root = tensor([r['root'] for r in records])[owner]
        root_q = tensor([r['root_quat'] for r in records])[owner]
        arm = tensor([r['arm'] for r in records])[owner]
        orientation = tensor([r['orientation'] for r in records])[owner]
        normal = tensor([r['normal'] for r in records])[owner]
        tangent = tensor([r['tangent'] for r in records])[owner]
        candidate = tensor([r['baseline'] for r in records])[owner]
        candidate += offset[:, :1] * normal + offset[:, 1:2] * tangent
        candidate[:, 2] += offset[:, 2]
        all_ik, all_inverse, errors = [], [], []
        compute_start = time.perf_counter()
        for low in range(0, len(bank), 2048):
            sl = slice(low, min(low + 2048, len(bank)))
            count = len(root[sl])
            solved, pos, rot = solve_candidate_ik(root[sl], root_q[sl], candidate[sl],
                quaternion_xyzw_to_matrix(orientation[sl][:, [1, 2, 3, 0]]), arm[sl],
                limits[:, 0][None].expand(count, -1), limits[:, 1][None].expand(count, -1),
                iterations=120, damping=.03)
            margin = torch.minimum(solved - limits[:, 0], limits[:, 1] - solved).amin(-1)
            ik = (pos <= 1.e-4) & (rot <= 2.e-4) & (margin >= .002)
            result = batched_steady_ee_inverse_no_gravity(candidate[sl], orientation[sl],
                base_position_world=root[sl], base_orientation_world_xyzw=root_q[sl],
                initial_arm_joint_positions=arm[sl], joint_limits=limits)
            ip, ir = result.position_error.norm(dim=-1), result.orientation_error.norm(dim=-1)
            all_ik.append(ik.cpu().numpy())
            all_inverse.append(((ip <= .010) & (ir <= 1.e-4)).cpu().numpy())
            errors.append(torch.stack((pos, rot, margin, ip, ir), -1).cpu().numpy())
        ik = np.concatenate(all_ik)
        inverse = np.concatenate(all_inverse)
        error = np.concatenate(errors)
        bank, owners = np.asarray(bank), np.asarray(owners)
        groups = {'all': np.ones(len(bank), bool),
                  'normal': (bank[:, 1:] == 0).all(1),
                  'normal_outward': (bank[:, 1:] == 0).all(1) & (bank[:, 0] >= 0),
                  'normal_inward': (bank[:, 1:] == 0).all(1) & (bank[:, 0] <= 0),
                  'tangent': (bank[:, [0, 2]] == 0).all(1),
                  'z': (bank[:, :2] == 0).all(1)}
        for distance in (-.05, -.10):
            fixed_normal = np.isclose(bank[:, 0], distance) & (bank[:, 1] == 0)
            groups[f'normal_{distance}_only'] = fixed_normal & (bank[:, 2] == 0)
            groups[f'normal_{distance}_with_height'] = fixed_normal
        for index, record in enumerate(records):
            record['refined'] = {}
            for name, mask in groups.items():
                ids = np.flatnonzero((owners == index) & mask & ik & inverse)
                selected = int(ids[np.argmin(np.linalg.norm(bank[ids], axis=1))]) if len(ids) else None
                record['refined'][name] = dict(offset_m=None if selected is None else bank[selected].tolist(),
                    distance_m=None if selected is None else float(np.linalg.norm(bank[selected])),
                    residuals=None if selected is None else error[selected].tolist())
            zero = int(np.flatnonzero((owners == index) & (bank == 0).all(1))[0])
            record['baseline_checks'] = dict(ik=bool(ik[zero]), inverse=bool(inverse[zero]),
                                             residuals=error[zero].tolist())
        report['refinement_compute_s'] = time.perf_counter() - compute_start
        report['refinement_candidates'] = len(bank)
        report['refined_summary'] = {name: dict(solved=sum(r['refined'][name]['distance_m'] is not None for r in records),
            median_distance_m=float(np.median([r['refined'][name]['distance_m'] for r in records
                                               if r['refined'][name]['distance_m'] is not None]))
            if any(r['refined'][name]['distance_m'] is not None for r in records) else None)
            for name in groups}
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: report[k] for k in ('refinement_compute_s', 'refinement_candidates', 'refined_summary')}, indent=2))
        return

    stage = Usd.Stage.Open(provenance['robot_asset']['path'])
    joints = {p.GetName(): p for p in stage.Traverse() if p.GetName().startswith('joint')}
    limits = tensor([[math.radians(joints[f'joint{i}'].GetAttribute('physics:lowerLimit').Get()),
                      math.radians(joints[f'joint{i}'].GetAttribute('physics:upperLimit').Get())]
                     for i in range(1, 7)])
    clouds = {}
    for body_name, key, tokens in ROBOT_COLLIDER_SPECS:
        if key in ('fixed_finger', 'moving_finger'):
            body = next(p for p in Usd.PrimRange(stage.GetDefaultPrim(), Usd.TraverseInstanceProxies())
                        if p.GetName().lower() == body_name.lower())
            clouds[key] = _read_collider_points(stage, body, tokens).points_body
    fixed = Rotation.from_euler('x', math.pi / 2).apply(clouds['fixed_finger'] - [.1, 0, 0])
    records = []
    for failure in failures:
        seed = failure['seed']
        shard = args.dataset / 'shards' / f'episode-seed{seed}'
        prov = json.loads((shard / 'meta/simulation_provenance.json').read_text())
        if prov['robot_asset']['path'] != provenance['robot_asset']['path']:
            raise ValueError('Mixed robot assets')
        episode = next(iter(prov['episodes'].values()))
        metadata = json.loads(next(args.dataset.glob(f'logs/rank*-control/seed_{seed}/metadata.json')).read_text())
        columns = ['frame_index', 'autoopen_stage', 'root_position_world', 'root_quaternion_xyzw',
                   'arm_joint_position', 'ee_pose_world_xyzw', 'gripper_feedback', 'door_hinge_position']
        rows = pq.read_table(next(shard.rglob('raw_audit/**/*.parquet')), columns=columns).to_pylist()
        row = next((r for r in rows if r['autoopen_stage'] == 'FAILURE'), rows[-1])
        # Canonical records contain the dataset tool frame: same point, +90deg x rotation.
        tool_rotation = Rotation.from_quat(row['ee_pose_world_xyzw'][3:]) * Rotation.from_euler('x', -math.pi / 2)
        initial_rotation = Rotation.from_quat(rows[0]['ee_pose_world_xyzw'][3:]) * Rotation.from_euler('x', -math.pi / 2)
        moving_initial = Rotation.from_euler('y', rows[0]['gripper_feedback']).apply(clouds['moving_finger'])
        separation = moving_initial.mean(0) - fixed.mean(0)
        separation /= np.linalg.norm(separation)
        depth = initial_rotation.inv().apply([-1., 0., 0.])
        depth -= separation * np.dot(depth, separation)
        depth /= np.linalg.norm(depth)
        prims = episode['door_asset']['prims']
        hinge_prim = next(p['attributes'] for p in prims if p['path'].endswith('/hinge_joint'))
        panel = next(p['attributes'] for p in prims if p['path'].endswith('/door_panel/panel'))
        origin = (np.asarray(metadata['robot_state']['initial_base_pose_w'][:3])
                  - np.asarray(metadata['robot_state']['initial_base_pose_env'][:3]))
        hinge = np.asarray(hinge_prim['physics:localPos0']) + origin
        center = np.asarray(panel['xformOp:translate']) + origin
        hq = hinge_prim.get('physics:localRot0', dict(real=1., imaginary=[0., 0., 0.]))
        sign = np.sign(Rotation.from_quat(hq['imaginary'] + [hq['real']]).apply([0., 0., 1.])[2])
        angle = sign * row['door_hinge_position']
        rotation = Rotation.from_euler('z', angle)
        normal = rotation.apply([1., 0., 0.])
        side = np.sign(np.dot(np.asarray(row['root_position_world']) - hinge, normal))
        normal *= side
        free = center - hinge
        free[[0, 2]] = 0
        free /= np.linalg.norm(free)
        tangent = rotation.apply(free)
        quat = tensor(tool_rotation.as_quat()[[3, 0, 1, 2]][None])
        orientation = align_vectors(_quat_rotate(quat, tensor(depth[None])), -tensor(normal[None]), quat)[0]
        moving = Rotation.from_euler('y', row['gripper_feedback']).apply(clouds['moving_finger'])
        offsets = _quat_rotate(orientation[None, None], tensor(np.concatenate((fixed, moving)))[None])
        minimum = float((offsets * tensor(normal)[None, None]).sum(-1).amin())
        root = np.asarray(row['root_position_world'])
        root_rotation = Rotation.from_quat(row['root_quaternion_xyzw'])
        yaw = root_rotation.as_euler('xyz')[2]
        anchor = root + .38 * np.asarray([math.cos(yaw), math.sin(yaw), 0.])
        distance = np.dot(anchor - hinge, tangent)
        surface = hinge + rotation.apply((center[0] - hinge[0]) * np.array([1., 0, 0])
                                        + panel['xformOp:scale'][0] * side * np.array([1., 0, 0]))
        surface += distance * tangent
        surface[2] = center[2] - panel['xformOp:scale'][2] + 1.
        baseline = surface + (.03 - minimum) * normal
        fk, _, _ = b2z1_gripper_pose_and_jacobian(tensor(root[None]),
            tensor(np.asarray(row['root_quaternion_xyzw'])[None]), tensor(np.asarray(row['arm_joint_position'])[None]))
        records.append(dict(seed=seed, frame=row['frame_index'], root=root.tolist(),
            root_quat=row['root_quaternion_xyzw'], arm=row['arm_joint_position'],
            baseline=baseline.tolist(), normal=normal.tolist(), tangent=tangent.tolist(),
            orientation=orientation.tolist(), door_deg=math.degrees(abs(angle)),
            prior_contacts=sum(r['autoopen_stage'] == 'CONTACT_PANEL' for r in rows),
            fk_logged_position_error_m=float((fk[0] - tensor(row['ee_pose_world_xyzw'][:3])).norm())))
    print(json.dumps({'loaded': len(records), 'load_s': time.perf_counter() - start,
                      'max_fk_record_error_m': max(r['fk_logged_position_error_m'] for r in records)}), flush=True)
    grid = np.asarray([(n, t, z) for n in np.arange(-.20, .201, args.step)
                       for t in np.arange(-.40, .401, args.step)
                       for z in np.arange(-.20, .201, args.step)])
    grid[np.abs(grid) < 1.e-8] = 0.
    grid = grid[np.argsort(np.linalg.norm(grid, axis=1), kind='stable')]
    offset = tensor(grid)
    groups = {'all': np.ones(len(grid), bool), 'normal': (grid[:, 1:] == 0).all(1),
              'tangent': (grid[:, [0, 2]] == 0).all(1), 'z': (grid[:, :2] == 0).all(1),
              'old_tangent_z': grid[:, 0] == 0, 'normal_z': grid[:, 1] == 0}
    compute_start = time.perf_counter()
    for index, record in enumerate(records):
        count = len(grid)
        base = tensor(record['root'])[None].expand(count, -1)
        base_q = tensor(record['root_quat'])[None].expand(count, -1)
        initial = tensor(record['arm'])[None].expand(count, -1)
        orientation = tensor(record['orientation'])[None].expand(count, -1)
        candidates = tensor(record['baseline'])[None] + offset[:, :1] * tensor(record['normal'])
        candidates += offset[:, 1:2] * tensor(record['tangent'])
        candidates[:, 2] += offset[:, 2]
        solved, pos, rot = solve_candidate_ik(base, base_q, candidates,
            quaternion_xyzw_to_matrix(orientation[:, [1, 2, 3, 0]]), initial,
            limits[:, 0][None].expand(count, -1), limits[:, 1][None].expand(count, -1),
            iterations=120, damping=.03)
        margin = torch.minimum(solved - limits[:, 0], limits[:, 1] - solved).amin(-1)
        ik_pass = (pos <= 1.e-4) & (rot <= 2.e-4) & (margin >= .002)
        result = batched_steady_ee_inverse_no_gravity(candidates, orientation,
            base_position_world=base, base_orientation_world_xyzw=base_q,
            initial_arm_joint_positions=initial, joint_limits=limits)
        ip = result.position_error.norm(dim=-1)
        ir = result.orientation_error.norm(dim=-1)
        inv_pass = (ip <= .010) & (ir <= 1.e-4)
        passed = (ik_pass & inv_pass).cpu().numpy()
        record['groups'] = {}
        for name, mask in groups.items():
            ids = np.flatnonzero(passed & mask)
            selected = int(ids[0]) if len(ids) else None
            record['groups'][name] = dict(count=int(len(ids)), offset_m=None if selected is None else grid[selected].tolist(),
                distance_m=None if selected is None else float(np.linalg.norm(grid[selected])),
                inverse_position_error_m=None if selected is None else float(ip[selected]),
                inverse_orientation_error_rad=None if selected is None else float(ir[selected]))
        record['ik_pass'] = int(ik_pass.sum())
        record['inverse_pass'] = int(inv_pass.sum())
        if (index + 1) % 10 == 0 or index == len(records) - 1:
            print(json.dumps({'completed': index + 1, 'total': len(records),
                              'compute_s': time.perf_counter() - compute_start}), flush=True)
    elapsed = time.perf_counter() - compute_start
    summary = {name: dict(solved=sum(r['groups'][name]['count'] > 0 for r in records),
        median_distance_m=float(np.median([r['groups'][name]['distance_m'] for r in records
                                          if r['groups'][name]['count']]))
        if any(r['groups'][name]['count'] for r in records) else None) for name in groups}
    output = dict(dataset=str(args.dataset), recorded_mount=recorded_mount, step_m=args.step,
        joint_limits=limits.tolist(),
        candidates_per_seed=len(grid), compute_s=elapsed, total_s=time.perf_counter() - start,
        reconstruction='First logged FAILURE row; orientation aligned from current actual EE. Obstacles ignored.',
        summary=summary, records=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({k: v for k, v in output.items() if k != 'records'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
