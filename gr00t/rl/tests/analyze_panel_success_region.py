"""Measure all accepted panel-contact visits in robot/door coordinates, offline."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('datasets', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    records = []
    for root in args.datasets:
        report = json.loads((root/'acceptance_report.json').read_text())
        positives = [e['seed'] for e in report['episodes'] if e['accepted']]
        for seed in positives:
            shard = root/'shards'/f'episode-seed{seed}'
            meta = json.loads(next(root.glob(f'logs/rank*-control/seed_{seed}/metadata.json')).read_text())
            provenance = json.loads((shard/'meta/simulation_provenance.json').read_text())
            prims = next(iter(provenance['episodes'].values()))['door_asset']['prims']
            hinge_attrs = next(p['attributes'] for p in prims if p['path'].endswith('/hinge_joint'))
            panel = next(p['attributes'] for p in prims if p['path'].endswith('/door_panel/panel'))
            origin = np.array(meta['robot_state']['initial_base_pose_w'][:3])-meta['robot_state']['initial_base_pose_env'][:3]
            hinge = origin+hinge_attrs['physics:localPos0']
            hq = hinge_attrs.get('physics:localRot0',dict(real=1.,imaginary=[0.,0.,0.]))
            sign = np.sign(Rotation.from_quat(hq['imaginary']+[hq['real']]).apply([0.,0.,1.])[2])
            free = origin+panel['xformOp:translate']-hinge
            free[[0,2]] = 0.
            free /= np.linalg.norm(free)
            columns = ['autoopen_stage','root_position_world','root_quaternion_xyzw',
                       'ee_inverse_input_world','ee_pose_world_xyzw','door_hinge_position']
            data = pq.read_table(next(shard.rglob('raw_audit/**/*.parquet')),columns=columns).to_pydict()
            stages = np.asarray(data['autoopen_stage'])
            entries = np.flatnonzero((stages[:-1]=='CONTACT_PANEL')&(stages[1:]=='PUSH_OPEN'))
            for frame in entries:
                robot = np.asarray(data['root_position_world'][frame])
                q = Rotation.from_quat(data['root_quaternion_xyzw'][frame])
                yaw = q.as_euler('xyz')[2]
                ee = np.asarray(data['ee_inverse_input_world'][frame])
                local = Rotation.from_euler('z',-yaw).apply(ee-robot)
                theta = data['door_hinge_position'][frame]
                tangent = Rotation.from_euler('z',sign*theta).apply(free)
                bottom = origin[2]+panel['xformOp:translate'][2]-panel['xformOp:scale'][2]
                records.append(dict(dataset=root.name,seed=seed,frame=int(frame),
                    hinge_sign=float(sign),door_angle_deg=float(np.degrees(theta)),
                    ee_base_x_m=float(local[0]),ee_base_y_m=float(local[1]),
                    ee_base_z_m=float(local[2]),ee_height_from_door_bottom_m=float(ee[2]-bottom),
                    ee_panel_radius_m=float(np.dot(ee-hinge,tangent)),
                    ee_shoulder_front_offset_m=float(local[0]-.48)))
        print(root.name,'accepted',len(positives),'visits',sum(r['dataset']==root.name for r in records),flush=True)
    keys = ('ee_base_x_m','ee_base_y_m','ee_base_z_m','ee_height_from_door_bottom_m',
            'ee_panel_radius_m','ee_shoulder_front_offset_m')
    groups = {'all':records}
    for sign in (-1.,1.):
        for low,high in ((0,45),(45,60),(60,75),(75,90)):
            groups[f'hinge{sign:+.0f}_angle{low}-{high}'] = [r for r in records if
                r['hinge_sign']==sign and low<=r['door_angle_deg']<high]
    summary = {name:dict(visits=len(rows),unique_seeds=len({r['seed'] for r in rows}),
        quantiles={key:np.quantile([r[key] for r in rows],[0,.05,.25,.5,.75,.95,1]).tolist() for key in keys})
        for name,rows in groups.items() if rows}
    args.output.write_text(json.dumps(dict(quantile_levels=[0,.05,.25,.5,.75,.95,1],
        datasets=[str(p) for p in args.datasets],summary=summary,records=records),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
