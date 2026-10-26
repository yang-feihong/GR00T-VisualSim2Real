"""Offline production-inverse and convex-collider wall checks; no simulator."""

import argparse
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
import torch

from gr00t.rl.auto_open.pull_station import collider_local_corners
from legged_gym.utils.ee_steady_inverse_gpu import batched_steady_ee_inverse_no_gravity


def transform(position, quaternion_xyzw):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    result[:3, 3] = position
    return result


def points_at(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def axes_unique(axes):
    axes = np.asarray(axes)
    axes = axes[np.linalg.norm(axes, axis=1) > 1.e-9]
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    return np.unique(np.round(np.concatenate((axes, -axes)), 8), axis=0)


def convex(points):
    hull = ConvexHull(points)
    edges = np.concatenate([points[hull.simplices[:, i]] - points[hull.simplices[:, j]]
                            for i, j in ((0, 1), (1, 2), (2, 0))])
    return dict(points=points[hull.vertices], normals=axes_unique(hull.equations[:, :3]),
                edges=axes_unique(edges))


def segment_box_distance(start, end, half):
    delta = end-start
    cuts = [0., 1.]
    for axis in range(3):
        if abs(delta[axis]) > 1.e-12:
            cuts.extend(float(t) for t in ((-half[axis]-start[axis])/delta[axis],
                                          (half[axis]-start[axis])/delta[axis]) if 0. < t < 1.)
    cuts = sorted(set(cuts))
    candidates = list(cuts)
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        mid = start + .5*(lo+hi)*delta
        active = np.abs(mid) > half
        bound = np.sign(mid)*half
        denom = np.dot(delta[active], delta[active])
        if denom > 1.e-20:
            optimum = np.dot(delta[active], bound[active]-start[active])/denom
            candidates.append(float(np.clip(optimum, lo, hi)))
    pts = start+np.asarray(candidates)[:, None]*delta
    return float(np.linalg.norm(np.maximum(np.abs(pts)-half, 0.), axis=1).min())


class Robot:
    def __init__(self, path):
        self.stage = Usd.Stage.Open(str(path))
        self.children = defaultdict(list)
        self.base = next(p for p in self.stage.Traverse()
                         if p.GetName() == 'base_link' and p.HasAPI(UsdPhysics.RigidBodyAPI))
        self.limits = []
        joints = {p.GetName(): p for p in self.stage.Traverse() if p.IsA(UsdPhysics.Joint)}
        for name in [f'joint{i}' for i in range(1, 7)]:
            j = joints[name]
            self.limits.append(np.deg2rad([j.GetAttribute('physics:lowerLimit').Get(),
                                          j.GetAttribute('physics:upperLimit').Get()]))
        for j in joints.values():
            a, b = (j.GetRelationship(f'physics:body{i}').GetTargets() for i in (0, 1))
            if not a or not b:
                continue
            frames = []
            for i in (0, 1):
                q = j.GetAttribute(f'physics:localRot{i}').Get()
                frames.append(transform(j.GetAttribute(f'physics:localPos{i}').Get(),
                                        [*q.GetImaginary(), q.GetReal()]))
            self.children[str(a[0])].append((j, str(b[0]), frames[0], np.linalg.inv(frames[1])))
        cache = UsdGeom.XformCache()
        self.shapes = []
        for body in self.stage.Traverse():
            if not body.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            if not (body.GetName().startswith('link') or body.GetName() in ('gripperMover', 'gripper_link')):
                continue
            inv = cache.GetLocalToWorldTransform(body).GetInverse()
            for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
                if not prim.HasAPI(UsdPhysics.CollisionAPI) or prim.GetAttribute('physics:collisionEnabled').Get() is False:
                    continue
                if prim.IsA(UsdGeom.Mesh):
                    pts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
                else:
                    pts = collider_local_corners(prim)
                xf = np.asarray(cache.GetLocalToWorldTransform(prim) * inv).T
                shape = convex(points_at(pts, xf))
                if prim.IsA(UsdGeom.Capsule):
                    cap = UsdGeom.Capsule(prim)
                    segment = np.zeros((2, 3))
                    segment[:, 'XYZ'.index(cap.GetAxisAttr().Get())] = np.array([-1., 1.])*cap.GetHeightAttr().Get()/2.
                    scales = np.linalg.norm(xf[:3, :3], axis=0)
                    if np.ptp(scales) > 1.e-6:
                        raise ValueError('Nonuniform capsule scale requires an ellipsoid check')
                    shape['capsule_segment'] = points_at(segment, xf)
                    shape['capsule_radius'] = float(cap.GetRadiusAttr().Get()*scales[0])
                self.shapes.append((str(body.GetPath()), str(prim.GetPath()), shape))

    def poses(self, root, arm, gripper):
        angles = dict(zip([f'joint{i}' for i in range(1, 7)], arm))
        angles['jointGripper'] = gripper
        base = str(self.base.GetPath())
        poses = {base: transform(root[:3], root[[4, 5, 6, 3]])}
        queue = deque([base])
        while queue:
            parent = queue.popleft()
            for j, child, f0, f1 in self.children[parent]:
                motion = np.eye(4)
                if j.IsA(UsdPhysics.RevoluteJoint):
                    motion[:3, :3] = Rotation.from_euler(j.GetAttribute('physics:axis').Get().lower(),
                                                        angles.get(j.GetName(), 0.)).as_matrix()
                elif j.IsA(UsdPhysics.PrismaticJoint):
                    motion['XYZ'.index(j.GetAttribute('physics:axis').Get()), 3] = angles.get(j.GetName(), 0.)
                poses[child] = poses[parent] @ f0 @ motion @ f1
                queue.append(child)
        return poses

    def check(self, root, arm, gripper, walls):
        poses = self.poses(root, arm, gripper)
        closest = None
        contacts = []
        for body, name, shape in self.shapes:
            xf = poses[body]
            pts = points_at(shape['points'], xf)
            normals = shape['normals'] @ xf[:3, :3].T
            edges = shape['edges'] @ xf[:3, :3].T
            for wall in walls:
                wp = wall['points']
                aabb_gap = np.maximum(wp.min(0) - pts.max(0), pts.min(0) - wp.max(0)).max()
                if aabb_gap > .05:
                    continue
                if 'capsule_segment' in shape:
                    segment = (points_at(shape['capsule_segment'], xf)-wall['center']) @ wall['rotation']
                    gap = segment_box_distance(segment[0], segment[1], wall['half'])-shape['capsule_radius']
                else:
                    axes = axes_unique(np.concatenate((normals, wall['normals'],
                        np.cross(edges[:, None, :], wall['edges'][None, :, :]).reshape(-1, 3))))
                    a, b = pts @ axes.T, wp @ axes.T
                    gap = np.maximum(b.min(0) - a.max(0), a.min(0) - b.max(0)).max()
                item = dict(body=body.rsplit('/', 1)[-1], collider=name, wall=wall['name'],
                            signed_sat_gap_m=float(gap))
                if closest is None or gap < closest['signed_sat_gap_m']:
                    closest = item
                if gap < -1.e-7:
                    contacts.append(item)
        return dict(collision=bool(contacts), contacts=contacts, closest=closest)


def walls_from_provenance(ep, offset):
    stage = Usd.Stage.CreateInMemory()
    for spec in ep['door_asset']['prims']:
        prim = stage.DefinePrim(spec['path'], spec['type'])
        for name, value in spec['attributes'].items():
            if name == 'xformOpOrder':
                prim.CreateAttribute(name, Sdf.ValueTypeNames.TokenArray).Set(value)
            elif name.startswith('xformOp:'):
                if isinstance(value, dict):
                    prim.CreateAttribute(name, Sdf.ValueTypeNames.Quatd).Set(
                        Gf.Quatd(value['real'], Gf.Vec3d(*value['imaginary'])))
                else:
                    prim.CreateAttribute(name, Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(*value))
            elif name == 'size':
                prim.GetAttribute(name).Set(value)
    cache = UsdGeom.XformCache()
    walls = []
    for spec in ep['door_asset']['prims']:
        if '/root_' not in spec['path'] or not spec['attributes'].get('physics:collisionEnabled'):
            continue
        prim = stage.GetPrimAtPath(spec['path'])
        if not prim.IsA(UsdGeom.Cube):
            continue
        xf = np.asarray(cache.GetLocalToWorldTransform(prim)).T
        pts = points_at(collider_local_corners(prim), xf) + offset
        shape = convex(pts)
        scale = np.linalg.norm(xf[:3, :3], axis=0)
        walls.append(dict(name=spec['path'], center=xf[:3, 3]+offset,
                          rotation=xf[:3, :3]/scale[None, :],
                          half=scale*float(UsdGeom.Cube(prim).GetSizeAttr().Get())/2., **shape))
    return walls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1967, 1968, 1969, 1970])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--paths', action='store_true')
    args = parser.parse_args()
    output = dict(dataset=str(args.dataset), method='Production CUDA inverse; USD joint FK; individual mesh convex hulls versus fixed wall/frame boxes by SAT; exact capsule-to-box distance for arm capsules. Mesh signed gap is a separating-axis bound, not Euclidean distance or PhysX penetration.', results=[])
    for seed in args.seeds:
        m = json.loads(next(args.dataset.glob(f'logs/rank*-control/seed_{seed}/metadata.json')).read_text())
        shard = args.dataset / 'shards' / f'episode-seed{seed}'
        provenance = json.loads((shard/'meta/simulation_provenance.json').read_text())
        asset = provenance['robot_asset']
        if hashlib.sha256(Path(asset['path']).read_bytes()).hexdigest() != asset['sha256']:
            raise ValueError('Robot asset differs from recorded asset')
        ep = next(iter(provenance['episodes'].values()))
        c = m['acceptance_diagnostics']['pull_panel_result']['cycles'][0]
        root = np.asarray(c['entry_root_state'])
        raw = pq.read_table(next(shard.rglob('raw_audit/**/*.parquet'))).to_pydict()
        stages = np.array(raw['autoopen_stage'])
        entry = int(np.flatnonzero(stages == 'INSERT_INTO_DOOR_GAP')[0])
        roots = np.array(raw['root_position_world'])
        ees = np.array(raw['ee_pose_world_xyzw'])
        match = np.linalg.norm(roots-root[:3], axis=1) + np.linalg.norm(ees[:, :3]-c['entry_ee_position_world_m'], axis=1)
        j = int(match.argmin())
        if match[j] > .01:
            raise ValueError(f'Entry mismatch {seed}: {match[j]}')
        arm = np.asarray(raw['arm_joint_position'][j])
        gripper = raw['gripper_feedback'][j]
        robot = Robot(asset['path'])
        offset = np.asarray(m['robot_state']['initial_base_pose_w'][:3])-m['robot_state']['initial_base_pose_env'][:3]
        walls = walls_from_provenance(ep, offset)
        k = c['selected_attempt']
        q = np.asarray(c['candidate_orientation_world_wxyz'][k])
        outward = -Rotation.from_quat(q[[1, 2, 3, 0]]).apply([1., 0., 0.])
        endpoint = np.asarray(c['candidate_endpoint_world_m'][k])
        alignment = endpoint + outward*(c['candidate_samples'][k][1]+.03)
        normal = np.array([1., 0., 0.])
        tangent = np.array([0., np.sign(root[1]-(offset[1])), 0.])
        variants = [(0., 0., z) for z in (0., .05, .10, .15, .20)]
        if args.sweep:
            variants = list(dict.fromkeys(variants + [(x, y, z) for x in (0., .025, .05, .075, .10, .15, .20)
                for y in (0., -.025, -.05, -.075, -.10, -.15, -.20)
                for z in (0., .05, .10, .15, .20)]))
        desired, tags = [], []
        for x, y, z in variants:
            delta = x*normal+y*tangent+np.array([0., 0., z])
            for name, point in [('alignment', alignment), ('endpoint', endpoint)]:
                desired.append(point+delta)
                tags.append((name, [x, y, z]))
        count = len(desired)
        def tensor(v):
            return torch.tensor(np.asarray(v), device='cuda:0', dtype=torch.float64)
        solved = batched_steady_ee_inverse_no_gravity(tensor(desired), tensor(np.tile(q, (count, 1))),
            base_position_world=tensor(np.tile(root[:3], (count, 1))),
            base_orientation_world_xyzw=tensor(np.tile(root[[4, 5, 6, 3]], (count, 1))),
            initial_arm_joint_positions=tensor(np.tile(arm, (count, 1))), joint_limits=tensor(robot.limits))
        joints = solved.arm_joint_positions.cpu().numpy()
        errors = np.linalg.norm(solved.position_error.cpu().numpy(), axis=-1)
        angles = 2*np.arcsin(np.clip(np.linalg.norm(solved.orientation_error.cpu().numpy(), axis=-1), 0, 1))
        result = dict(seed=seed, entry_match_m=float(match[j]), entry_frame=entry,
            root=root.tolist(), entry_arm=arm.tolist(), orientation_wxyz=q.tolist(),
            alignment_world=alignment.tolist(), endpoint_world=endpoint.tolist(), trials=[])
        for i, (name, delta) in enumerate(tags):
            check = robot.check(root, joints[i], gripper, walls)
            result['trials'].append(dict(pose=name, delta_normal_tangent_z_m=delta,
                target_world=desired[i].tolist(), inverse_position_error_m=float(errors[i]),
                inverse_orientation_error_rad=float(angles[i]), arm_joints_rad=joints[i].tolist(), **check))
        actual_fk = robot.poses(root, arm, gripper)
        result['entry_fk_link06_world'] = next(v[:3, 3].tolist() for p, v in actual_fk.items() if p.endswith('/link06'))
        result['entry_recorded_ee_world'] = ees[j, :3].tolist()
        ee_xf = next(v for p, v in actual_fk.items() if p.endswith('/gripper_link'))
        result['entry_fk_ee_error_m'] = float(np.linalg.norm(ee_xf[:3, 3]-ees[j, :3]))
        fk_errors = []
        for i, target in enumerate(desired):
            xf = next(v for p, v in robot.poses(root, joints[i], gripper).items() if p.endswith('/gripper_link'))
            fk_errors.append(float(np.linalg.norm(xf[:3, 3]-target)))
        result['target_asset_fk_error_m'] = fk_errors
        if args.paths:
            fraction = np.linspace(0., 1., 81)
            start_q = Rotation.from_matrix(ee_xf[:3, :3]).as_quat()
            q_path = Slerp([0., 1.], Rotation.from_quat([start_q, q[[1, 2, 3, 0]]]))(fraction).as_quat()[:, [3, 0, 1, 2]]
            path_tags, path_positions, path_orientations = [], [], []
            for dz in (0., .05, .10, .15, .20):
                goal = alignment + [0., 0., dz]
                pts = ee_xf[:3, 3] + fraction[:, None]*(goal-ee_xf[:3, 3])
                path_tags.append(dz)
                path_positions.extend(pts)
                path_orientations.extend(q_path)
            n = len(path_positions)
            path_inverse = batched_steady_ee_inverse_no_gravity(tensor(path_positions), tensor(path_orientations),
                base_position_world=tensor(np.tile(root[:3], (n, 1))),
                base_orientation_world_xyzw=tensor(np.tile(root[[4, 5, 6, 3]], (n, 1))),
                initial_arm_joint_positions=tensor(np.tile(arm, (n, 1))), joint_limits=tensor(robot.limits))
            path_joints = path_inverse.arm_joint_positions.cpu().numpy().reshape(-1, 81, 6)
            residuals = np.linalg.norm(path_inverse.position_error.cpu().numpy(), axis=-1).reshape(-1, 81)
            result['offline_paths'] = []
            for row, dz in enumerate(path_tags):
                checks = [robot.check(root, v, gripper, walls) for v in path_joints[row]]
                collisions = [dict(fraction=float(fraction[i]), **v) for i, v in enumerate(checks) if v['collision']]
                result['offline_paths'].append(dict(kind='Cartesian straight line with quaternion Slerp; each point uses entry-state production inverse, not a dynamics replay', dz_m=dz,
                    max_inverse_position_error_m=float(residuals[row].max()), collision_samples=len(collisions),
                    first_collision=collisions[0] if collisions else None,
                    worst_gap_m=min((v['closest']['signed_sat_gap_m'] for v in checks if v['closest']), default=None)))
            print('PATHS', seed, [(p['dz_m'],p['collision_samples'],p['worst_gap_m']) for p in result['offline_paths']], flush=True)
        report = json.loads((shard/'meta/acceptance_report.json').read_text())['episodes'][0]
        contacts = [v for v in report['contacts']['abnormal_contacts'] if v['stage']=='INSERT_INTO_DOOR_GAP']
        if contacts:
            first = min(contacts, key=lambda v:v['frame'])
            row = raw['frame_index'].index(first['frame'])
            r = np.r_[raw['root_position_world'][row], np.array(raw['root_quaternion_xyzw'][row])[[3, 0, 1, 2]]]
            result['recorded_first_contact'] = first
            result['recorded_contact_pose_check'] = robot.check(r, raw['arm_joint_position'][row], raw['gripper_feedback'][row], walls)
            result['recorded_contact_arm_joints_rad'] = raw['arm_joint_position'][row]
            result['recorded_contact_arm_target_rad'] = raw['arm_joint_position_target'][row]
            result['recorded_contact_target_pose_check'] = robot.check(r, raw['arm_joint_position_target'][row], raw['gripper_feedback'][row], walls)
        output['results'].append(result)
        initial = [r for r in result['trials'] if r['delta_normal_tangent_z_m'][:2] == [0., 0.]]
        print(seed, [(r['pose'], r['delta_normal_tangent_z_m'][2], round(r['inverse_position_error_m']*1000, 3),
                      [(s['body'], s['wall'].rsplit('/',1)[-1], round(s['signed_sat_gap_m']*1000, 2)) for s in r['contacts']]) for r in initial], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print('OUTPUT', args.output)


if __name__ == '__main__':
    main()
