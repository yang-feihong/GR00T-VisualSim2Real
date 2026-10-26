"""Validate the current region sampler on recorded successes or failures, without simulation."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
import torch

from gr00t.rl.auto_open.geometry import _read_collider_points
from gr00t.rl.auto_open.tensor_planning import TensorPlanningCoordinator
from gr00t.rl.auto_open.tensor_stage_planning import align_vectors, _quat_rotate
from gr00t.rl.auto_open import panel_push_forecast as forecast_module
from gr00t.rl.scripts.probe_auto_open_arm_wall import Robot, points_at
from gr00t.rl.envs.door.b2z1_reset_geometry import b2z1_gripper_pose_and_jacobian


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--seeds',type=int,nargs='+')
    parser.add_argument('--path-gap-audit',action='store_true')
    parser.add_argument('--joint-first',type=int,choices=range(6))
    parser.add_argument('--cohort', choices=['successful','failed','runtime_failed','arrival_failed'], default='successful')
    parser.add_argument('--arrival', action='store_true')
    parser.add_argument('--terminal-audit', action='store_true',
                        help='Replay arrival and terminal inverse at recorded base poses, without simulation.')
    parser.add_argument('--anchor', choices=['pull','push'], default='pull',
                        help='Pull stance-relative anchor or unchanged push 0.40 m anchor.')
    parser.add_argument('--records-from', type=Path,
                        help='Reuse reconstructed offline inputs from an earlier report.')
    parser.add_argument('--selection', choices=['pull','push'], default='pull')
    parser.add_argument('--chunk-size', type=int, default=16)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.joint_first is not None:
        import inspect
        import gr00t.rl.envs.door.b2z1_reset_geometry as arm_geometry
        original_fk = arm_geometry.b2z1_gripper_pose_and_jacobian
        def joint_first_fk(base,quat,arms):
            if inspect.currentframe().f_back.f_code.co_name=='validate_panel_arrivals':
                path = arms.reshape(-1,384,6)
                start,end = path[:,0].clone(),path[:,-1].clone()
                delta = end-start
                fraction = ((path-start[:,None])*delta[:,None]).sum(-1)/delta.square().sum(-1)[:,None].clamp_min(1.e-12)
                first_distance = delta[:,args.joint_first].abs()
                other = delta.abs().clone()
                other[:,args.joint_first] = 0.
                share = first_distance/(first_distance+other.amax(-1)).clamp_min(1.e-12)
                first = (fraction/share[:,None].clamp_min(1.e-12)).clamp(0.,1.)
                rest = ((fraction-share[:,None])/(1-share)[:,None].clamp_min(1.e-12)).clamp(0.,1.)
                factors = rest[...,None].expand_as(path).clone()
                factors[:,:,args.joint_first] = first
                path.copy_(start[:,None]+factors*delta[:,None])
            return original_fk(base,quat,arms)
        arm_geometry.b2z1_gripper_pose_and_jacobian = joint_first_fk
    if args.terminal_audit:
        args.arrival = True
    report = json.loads((args.dataset/'acceptance_report.json').read_text())
    seeds = [e['seed'] for e in report['episodes'] if e['accepted']] if args.cohort=='successful' else [
        e['seed'] for e in report['episodes'] if not e['accepted'] and
        e['reasons'][0] == 'autoopen_failure:CONTACT_PANEL:panel_contact_inverse_mapping_failed']
    if args.cohort in ('runtime_failed','arrival_failed'):
        seeds = sorted({failure['seed']
            for path in args.dataset.glob('logs/rank*-control/summary.json')
            for failure in json.loads(path.read_text())['failures']
            if failure['stage'] == 'CONTACT_PANEL'
            and failure['reason'] == ('contact_panel_acceptance_timeout'
                if args.cohort=='arrival_failed' else 'panel_contact_inverse_mapping_failed')})
    if args.seeds:
        seeds = [seed for seed in seeds if seed in args.seeds]
    if args.limit:
        seeds = seeds[:args.limit]
    if args.records_from:
        seeds = []
    records = []
    tensor = lambda value: torch.as_tensor(np.asarray(value), device=args.device, dtype=torch.float64)
    robot = None
    for seed in seeds:
        meta = json.loads(next(args.dataset.glob(f'logs/rank*-control/seed_{seed}/metadata.json')).read_text())
        shard = args.dataset/'shards'/f'episode-seed{seed}'
        prov = json.loads((shard/'meta/simulation_provenance.json').read_text())
        if robot is None:
            robot = Robot(prov['robot_asset']['path'])
            shapes = []
            for body, tokens in [('link06', ('fixed_convex_023', 'fixed_convex_024')),
                                 ('gripperMover', ('moving_convex_000',))]:
                prim = next(p for p in robot.stage.Traverse() if p.GetName() == body)
                shapes.append((str(prim.GetPath()), _read_collider_points(robot.stage, prim, tokens).points_body))
        d = pq.read_table(next(shard.rglob('raw_audit/**/*.parquet')), columns=[
            'autoopen_stage', 'root_position_world', 'root_quaternion_xyzw', 'ee_pose_world_xyzw',
            'arm_joint_position', 'gripper_feedback', 'door_hinge_position', 'handle_position']).to_pydict()
        if args.cohort in ('successful','arrival_failed'):
            contact = np.flatnonzero(np.asarray(d['autoopen_stage'])=='CONTACT_PANEL')
            if not len(contact):
                continue
            i = np.split(contact,np.flatnonzero(np.diff(contact)>1)+1)[-1][0]
            if args.cohort=='arrival_failed' or args.arrival:
                i = max(0,i-1)
        else:
            i = next((j for j, stage in enumerate(d['autoopen_stage']) if stage == 'FAILURE'), len(d['autoopen_stage'])-1)
            if args.cohort=='runtime_failed':
                i = max(0,i-1)
        def clouds(j):
            q = np.asarray(d['root_quaternion_xyzw'][j])
            poses = robot.poses(np.r_[d['root_position_world'][j], q[[3,0,1,2]]],
                                d['arm_joint_position'][j], d['gripper_feedback'][j])
            return [points_at(points, poses[body]) for body, points in shapes]
        ee = np.asarray(d['ee_pose_world_xyzw'])
        ee_q = Rotation.from_quat(ee[:,3:])*Rotation.from_euler('x', -math.pi/2)
        initial_local = [ee_q[0].inv().apply(cloud-ee[0,:3]) for cloud in clouds(0)]
        separation = initial_local[1].mean(0)-initial_local[0].mean(0)
        separation /= np.linalg.norm(separation)
        normal_closed = np.array([1. if meta['scene']['robot_side']=='positive_x' else -1., 0., 0.])
        depth = ee_q[0].inv().apply(-normal_closed)
        depth -= separation*np.dot(depth, separation)
        depth /= np.linalg.norm(depth)
        local = ee_q[i].inv().apply(np.concatenate(clouds(i))-ee[i,:3])
        prims = next(iter(prov['episodes'].values()))['door_asset']['prims']
        attrs = lambda suffix: next(p['attributes'] for p in prims if p['path'].endswith(suffix))
        ha, panel = attrs('/hinge_joint'), attrs('/door_panel/panel')
        origin = np.array(meta['robot_state']['initial_base_pose_w'][:3])-meta['robot_state']['initial_base_pose_env'][:3]
        hinge = origin+ha['physics:localPos0']
        hq = ha.get('physics:localRot0', dict(real=1., imaginary=[0.,0.,0.]))
        sign = np.sign(Rotation.from_quat(hq['imaginary']+[hq['real']]).apply([0.,0.,1.])[2])
        theta = d['door_hinge_position'][i]
        rotation = Rotation.from_euler('z', sign*theta)
        center_closed = origin+panel['xformOp:translate']
        center = hinge+rotation.apply(center_closed-hinge)
        normal = rotation.apply(normal_closed)
        normal *= np.sign(np.dot(np.asarray(d['root_position_world'][i])-center, normal))
        free = center_closed-hinge
        free[[0,2]] = 0.
        free /= np.linalg.norm(free)
        tangent = rotation.apply(free)
        initial = meta['acceptance_diagnostics'].get('panel_grid_inverse')
        seed_q = ee_q[i]
        previous_push = np.flatnonzero((np.asarray(d['autoopen_stage'])=='PUSH_OPEN') & (np.arange(len(ee))<i))
        if len(previous_push):
            previous_push = np.split(previous_push,np.flatnonzero(np.diff(previous_push)>1)+1)[-1]
            push_entry = max(0,int(previous_push[0])-1)
            delta = sign*(theta-d['door_hinge_position'][push_entry])
            seed_q = Rotation.from_euler('z',delta)*ee_q[push_entry]
        q = tensor(seed_q.as_quat()[[3,0,1,2]])[None]
        orientation = align_vectors(_quat_rotate(q,tensor(depth)[None]), -tensor(normal)[None],q).float()
        offsets = _quat_rotate(orientation[:,None],tensor(local)[None].float())
        yaw = Rotation.from_quat(d['root_quaternion_xyzw'][i]).as_euler('xyz')[2]
        nominal = np.dot(np.asarray(d['root_position_world'][i])+ .38*np.array([math.cos(yaw),math.sin(yaw),0.])-hinge,tangent)
        if args.anchor == 'push':
            nominal = .40
        radius = nominal
        surface = hinge+radius*tangent+(np.dot(center-hinge,normal)+panel['xformOp:scale'][0])*normal
        surface[2] = center_closed[2]-panel['xformOp:scale'][2]+1.
        desired = tensor(surface)[None]+(.03-(offsets*tensor(normal)[None,None]).sum(-1).amin(-1))[:,None]*tensor(normal)[None]
        hj, hc = attrs('/handle_joint'), attrs('/handle_outside')
        handle_q = hj.get('physics:localRot0',dict(real=1.,imaginary=[0.,0.,0.]))
        hs = np.sign(Rotation.from_quat(handle_q['imaginary']+[handle_q['real']]).apply([1.,0.,0.])[0])
        handle_rotation = Rotation.from_euler('x',hs*d['handle_position'][i])
        axle = hinge+rotation.apply(origin+hj['physics:localPos0']-hinge)
        # Door handle capsule uses its local Y axis; asset center is in handle-body coordinates.
        handle_local = np.array(hc.get('xformOp:translate',[0.,0.,0.]))
        handle_center = axle+rotation.apply(handle_rotation.apply(handle_local))
        axis = rotation.apply(handle_rotation.apply([0.,-1.,0.]))
        half_height = abs(axis[2])*hc['height']/2+hc['radius']
        rq = np.array(d['root_quaternion_xyzw'][i])
        root = np.r_[d['root_position_world'][i],rq[[3,0,1,2]]]
        fk, fk_rotation, _ = b2z1_gripper_pose_and_jacobian(tensor(root[:3])[None], tensor(rq)[None],tensor(d['arm_joint_position'][i])[None])
        records.append(dict(seed=seed, root=root.tolist(), arm=d['arm_joint_position'][i],
            hinge=hinge.tolist(), sign=float(sign), theta=theta, depth=depth.tolist(), jaw_local=local.tolist(),
            normal=normal.tolist(), tangent=tangent.tolist(), desired=desired[0].tolist(), orientation=orientation[0].tolist(),
            edge=float(np.dot(center_closed-hinge,free)+panel['xformOp:scale'][1]),
            bottom=float(center_closed[2]-panel['xformOp:scale'][2]),top=float(center_closed[2]+panel['xformOp:scale'][2]),
            handle_bounds=[handle_center[2]-half_height,handle_center[2]+half_height],
            recorded=meta['acceptance_diagnostics'].get('panel_push_forecast_inverse'),
            ee_position=ee[i,:3].tolist(),ee_orientation=ee_q[i].as_quat()[[3,0,1,2]].tolist(),
            gripper=float(d['gripper_feedback'][i]),
            static_point_present=initial is not None, nominal_radius_m=float(nominal),
            actual_fk_error_m=float(torch.linalg.vector_norm(fk[0]-tensor(ee[i,:3])))))
    if args.records_from:
        records = json.loads(args.records_from.read_text())['records']
        if args.limit:
            records = records[:args.limit]
        for record in records:
            root = np.asarray(record['root'])
            yaw = Rotation.from_quat(root[[4,5,6,3]]).as_euler('xyz')[2]
            nominal = (.40 if args.anchor == 'push' else np.dot(
                root[:3] + .38*np.array([math.cos(yaw),math.sin(yaw),0.])
                - record['hinge'], record['tangent']))
            record['desired'] = (np.asarray(record['desired']) +
                (nominal-record['nominal_radius_m'])*np.asarray(record['tangent'])).tolist()
            record['nominal_radius_m'] = float(nominal)
        prov = json.loads((args.dataset/'shards'/f"episode-seed{records[0]['seed']}"/
                           'meta/simulation_provenance.json').read_text())
        robot = Robot(prov['robot_asset']['path'])
    n = len(records)
    coordinator = object.__new__(TensorPlanningCoordinator)
    coordinator.arm_joint_ids = torch.arange(6,device=args.device)
    coordinator.joint_limits = tensor(robot.limits)
    coordinator.path_diagnostics = []
    coordinator.gpu_stage_constants = dict(
        retract_lower=tensor(robot.limits)[:,0].expand(n,-1),
        retract_upper=tensor(robot.limits)[:,1].expand(n,-1),
        panel_standoff_target_clearance=torch.full((n,),.03,device=args.device),
        panel_contact_edge_margin=torch.full((n,),.05,device=args.device),
        panel_hinge_to_free_edge=tensor([r['edge'] for r in records]),
        panel_bottom_z=tensor([r['bottom'] for r in records]),panel_top_z=tensor([r['top'] for r in records]))
    fsm = SimpleNamespace(device=args.device, hinge_idx=0, pull_door=args.selection=='pull',
        hinge_axis_sign=tensor([r['sign'] for r in records]).float(),hinge_world=tensor([r['hinge'] for r in records]).float(),
        depth_axis_local=tensor([r['depth'] for r in records]).float(),push_comfort_reach_radius=torch.full((n,),.7,device=args.device),
        env=SimpleNamespace(get_goal_center_offset_local=lambda: tensor([[.2,0.,.7]]*n)),
        push_planned_stop_hinge=torch.zeros(n,device=args.device),
        panel_push_tip_offset_local=torch.zeros((n,3),device=args.device),
        panel_push_tip_surface_world=torch.zeros((n,3),device=args.device),
        panel_push_tip_hinge_angle=torch.zeros(n,device=args.device),fail=lambda rows,code:None)
    state = dict(root_states=tensor([r['root'] for r in records]).float(),dof_pos=tensor([r['arm'] for r in records]).float(),
                 door_joint_position=tensor([[r['theta']] for r in records]).float())
    gap_audit = []
    if args.path_gap_audit:
        from gr00t.rl.auto_open.first_pull_wall import FirstPullArmWallGPU
        envelope = FirstPullArmWallGPU(device=args.device)
        coordinator._panel_arrival_envelope = envelope
        original_inverse = coordinator._inverse_pose_bank
        original_clearance = envelope.clearance
        latest = {}
        def inverse(*values,**kwargs):
            result,owner,sample = original_inverse(*values,**kwargs)
            latest.update(owner=owner,sample=sample,lengths=values[5])
            return result,owner,sample
        def clearance(*values,**kwargs):
            gap = original_clearance(*values,**kwargs)
            owner,sample,lengths = latest['owner'],latest['sample'],latest['lengths']
            end = sample==lengths[owner]-1
            first = sample==0
            item = dict(paths=len(lengths),sample_count=len(gap),
                first_min_gap_m=float(gap[first].min()),end_min_gap_m=float(gap[end].min()),
                end_clear_count=int((gap[end]>=.003).sum()),
                minimum_gap_m=float(gap.min()))
            minimum = torch.full((len(lengths),),torch.inf,device=gap.device,dtype=gap.dtype)
            minimum.scatter_reduce_(0,owner,gap,reduce='amin')
            joint_paths = minimum[::2]
            item['best_joint_route_min_gap_m'] = float(joint_paths[torch.isfinite(joint_paths)].max())
            gap_audit.append(item)
            return gap
        coordinator._inverse_pose_bank = inverse
        envelope.clearance = clearance
    if args.arrival:
        from gr00t.rl.auto_open.tensor_fsm import PaddedPathBank
        fsm.panel_arrival_path = PaddedPathBank(n,384,args.device)
        state.update(ee_pos=tensor([r['ee_position'] for r in records]).float(),
            ee_orn=tensor([r['ee_orientation'] for r in records]).float(),
            gripper_position=tensor([[r['gripper']] for r in records]).float())
    select = forecast_module.select_panel_arc
    capture = {}
    def capture_select(valid,lengths,angles,geometry,**runtime_options):
        options = runtime_options
        _, accepted, stop, complete = select(valid.reshape(-1,1,valid.shape[-1]),
            lengths.reshape(-1,1),angles.reshape(-1,1,angles.shape[-1]),geometry.reshape(-1,1),**options)
        capture.update(feasible=accepted.reshape(len(valid),-1).sum(-1).cpu().tolist())
        samples = torch.arange(valid.shape[-1], device=valid.device)
        first_bad = torch.where((samples[None,None] < lengths[...,None]) & ~valid,
                                samples, valid.shape[-1]).amin(-1)
        complete = first_bad >= lengths
        last = torch.minimum((first_bad-2).clamp(0,angles.shape[-1]-1),
                             (lengths-2).clamp_min(0))
        progress = angles.gather(-1,last[...,None]).squeeze(-1)-angles[...,0]
        progress -= (~complete)*forecast_module.PUSH_TIP_MAX_ANGULAR_LEAD_RAD
        comfort_progress = angles.gather(-1,(lengths-2).clamp_min(0)[...,None]).squeeze(-1)-angles[...,0]
        capture.update(best_progress_deg=torch.rad2deg(torch.where(
            geometry,progress,torch.full_like(progress,-torch.inf)).amax(-1)).cpu().tolist(),
            best_comfort_progress_deg=torch.rad2deg(torch.where(
                geometry,comfort_progress,torch.full_like(progress,-torch.inf)).amax(-1)).cpu().tolist())
        return select(valid,lengths,angles,geometry,**options)
    forecast_module.select_panel_arc = capture_select
    torch.cuda.synchronize()
    started = time.perf_counter()
    for begin in range(0,n,args.chunk_size):
        end = min(begin+args.chunk_size,n)
        chunk = records[begin:end]
        coordinator._forecast_panel_push(fsm,state,torch.arange(begin,end,device=args.device),
            tensor([r['desired'] for r in chunk]).float(),tensor([r['orientation'] for r in chunk]).float(),
            tensor([r['normal'] for r in chunk]).float(),tensor([r['tangent'] for r in chunk]).float(),
            tensor([r['jaw_local'] for r in chunk]).float(),tensor([r['handle_bounds'] for r in chunk]).float())
        diagnostics = {k:v.cpu().tolist() for k,v in coordinator.path_diagnostics[-1][2].items()}
        arrival_diagnostics = ({k:v.cpu().tolist() for k,v in coordinator.path_diagnostics[-2][2].items()}
                               if args.arrival else None)
        for j,r in enumerate(chunk):
            r['replayed'] = {k:v[j] for k,v in diagnostics.items()}
            if args.arrival:
                r['arrival'] = {k:v[j] for k,v in arrival_diagnostics.items()}
                last = int(fsm.panel_arrival_path.length[begin+j])-1
                if last >= 0:
                    r['arrival_endpoint_world'] = fsm.panel_arrival_path.positions[begin+j,last].cpu().tolist()
                    r['arrival_orientation_world_wxyz'] = fsm.panel_arrival_path.orientations[begin+j,last].cpu().tolist()
                if args.terminal_audit and last >= 0:
                    from gr00t.rl.tests.debug_panel_terminal_inverse import audit_terminal
                    r['terminal_inverse_audit'] = audit_terminal(
                        args.dataset,r,coordinator.joint_limits,args.device)
            if not args.arrival:
                r['feasible_candidates'] = capture['feasible'][j]
                r['best_progress_deg'] = capture['best_progress_deg'][j]
                r['best_comfort_progress_deg'] = capture['best_comfort_progress_deg'][j]
        print('processed',end,'/',n,flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-started
    summary = dict(seeds=n,candidates_per_scene=len(records[0]['replayed']['candidate_geometry_valid']),compute_s=elapsed,
        solved=sum(r['replayed']['accepted'] for r in records),
        cohort=args.cohort,
        anchor=args.anchor,
        selection=args.selection,
        original_recorded_rejections=sum(r['recorded'] is not None and not r['recorded']['accepted'] for r in records),
        actual_fk_error_max_m=max(r['actual_fk_error_m'] for r in records),
        rejected_seeds=[r['seed'] for r in records if not r['replayed']['accepted']])
    args.output.write_text(json.dumps(dict(summary=summary,records=records,path_gap_audit=gap_audit),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
