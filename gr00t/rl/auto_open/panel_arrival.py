"""Batched stage-entry panel arrival paths, clear of the panel half-space."""
import torch


def panel_standing_height_preference(coordinator,fsm,state,rows,bank):
    from legged_gym.utils.ee_steady_inverse_gpu import batched_steady_ee_inverse_no_gravity

    roots = state['root_states'][rows].double()
    rewards = getattr(getattr(fsm.env,'cfg',None),'rewards',None)
    nominal_height = float(getattr(rewards,'base_height_target',.55))
    # The whole-body policy can rise while holding a height-invariant EE goal.
    upper = torch.maximum(roots[:,2],roots.new_full((len(rows),),nominal_height+.05))
    heights = roots[:,2,None]+roots.new_tensor([.5,1.])[None]*(upper-roots[:,2])[:,None]
    n,c = bank['targets'].shape[:2]
    base = roots[:,None,None].expand(-1,c,2,-1).clone()
    base[:,:,:,2] = heights[:,None]
    result = batched_steady_ee_inverse_no_gravity(
        bank['targets'][:,:,None].expand(-1,-1,2,-1).flatten(0,2),
        bank['quats'][:,:,None].expand(-1,-1,2,-1).flatten(0,2),
        base_position_world=base[:,:,:,:3].flatten(0,2),
        base_orientation_world_xyzw=base[:,:,:,[4,5,6,3]].flatten(0,2),
        initial_arm_joint_positions=bank['entry_joints'][:,:,None].expand(-1,-1,2,-1).flatten(0,2),
        joint_limits=coordinator.joint_limits)
    limits = coordinator.joint_limits.to(result.arm_joint_positions)
    margin = torch.minimum(result.arm_joint_positions-limits[:,0],limits[:,1]-result.arm_joint_positions).amin(-1)
    good = ((result.position_error.norm(dim=-1)<=.001)
        & (result.orientation_error.norm(dim=-1)<=.005) & (margin>=.05))
    return good.reshape(n,c,2).all(-1)

def panel_arrival_path(start, start_q, end, end_q, normal, jaw_local, *, capacity=400):
    from gr00t.rl.auto_open.tensor_stage_planning import _slerp

    # Rotate and translate laterally only where every tool attitude fits.
    radius = torch.linalg.vector_norm(jaw_local,dim=-1).amax(-1)+.03
    start_n = ((start-end)*normal).sum(-1)
    safe_n = torch.maximum(start_n,radius)
    staging = start+(safe_n-start_n)[:,None]*normal
    aligned = end+radius[:,None]*normal
    points = torch.stack((start,staging,aligned,end),1)
    quats = torch.stack((start_q,start_q,end_q,end_q),1)
    distance = torch.linalg.vector_norm(points[:,1:]-points[:,:-1],dim=-1)
    angle = 2*torch.acos((quats[:,1:]*quats[:,:-1]).sum(-1).abs().clamp(0.,1.))
    steps = torch.maximum(torch.ceil(distance/.003),torch.ceil(angle/.025)).long()
    ends = steps.cumsum(-1)
    lengths = ends[:,-1]+1
    ticks = torch.arange(capacity,device=start.device)[None].expand(len(start),-1)
    ticks = torch.minimum(ticks,(lengths-1)[:,None])
    segment = torch.searchsorted(ends.contiguous(),ticks.contiguous(),right=False).clamp_max(2)
    previous = torch.nn.functional.pad(ends[:,:-1],(1,0)).gather(1,segment)
    fraction = (ticks-previous).to(start.dtype)/steps.gather(1,segment).clamp_min(1)
    gather = lambda values,index:values.gather(1,index[...,None].expand(-1,-1,values.shape[-1]))
    p0,p1 = gather(points,segment),gather(points,segment+1)
    q0,q1 = gather(quats,segment),gather(quats,segment+1)
    positions = p0+(p1-p0)*fraction[...,None]
    orientations = _slerp(q0,q1,fraction)
    positions[:,0],orientations[:,0] = start,start_q
    return positions,orientations,lengths


def panel_joint_route(start,end,capacity,*,yaw_first=False):
    delta = end-start
    tick = torch.arange(capacity,device=start.device)[None]
    if not yaw_first:
        steps = torch.ceil(delta.abs().amax(-1)/.02).long().clamp_min(1)
        fraction = (tick/steps[:,None]).clamp_max(1.)
        return start[:,None]+fraction[...,None]*delta[:,None],steps+1
    tick = tick.to(start.dtype)
    first = torch.ceil(delta[:,0].abs()/.02).long()
    rest = torch.ceil(delta[:,1:].abs().amax(-1)/.02).long()
    count = (first+rest).clamp_min(1)+1
    first_fraction = (tick/first[:,None].clamp_min(1)).clamp_max(1.)
    rest_fraction = ((tick-first[:,None])/rest[:,None].clamp_min(1)).clamp(0.,1.)
    fraction = rest_fraction[...,None].expand(-1,-1,6).clone()
    fraction[:,:,0] = first_fraction
    return start[:,None]+fraction*delta[:,None],count


def validate_panel_arrivals(coordinator,fsm,state,rows,bank,valid,lengths,angles,candidate_valid,
                            *,_yaw_first=False):
    from gr00t.rl.auto_open.panel_push_forecast import select_panel_arc
    from gr00t.rl.auto_open.first_pull_wall import FirstPullArmWallGPU,retime_first_pull_paths
    from gr00t.rl.envs.door.b2z1_reset_geometry import b2z1_gripper_pose_and_jacobian
    from gr00t.rl.auto_open.pull_entry_gpu import _matrix_quat

    if not hasattr(coordinator,'_panel_arrival_envelope'):
        coordinator._panel_arrival_envelope = FirstPullArmWallGPU(device=fsm.device)
    capacity = fsm.panel_arrival_path.capacity
    available = candidate_valid.clone()
    preferred = panel_standing_height_preference(coordinator,fsm,state,rows,bank) & candidate_valid
    _,has_arc,_,_ = select_panel_arc(valid.flatten(0,1)[:,None],lengths.flatten()[:,None],
        angles.flatten(0,1)[:,None],candidate_valid.flatten()[:,None],minimum_progress_rad=0.)
    preferred &= has_arc.reshape_as(preferred)
    accepted = torch.zeros(len(rows),device=fsm.device,dtype=torch.bool)
    chosen = torch.zeros(len(rows),device=fsm.device,dtype=torch.int64)
    selected_stop = torch.zeros(len(rows),device=fsm.device,dtype=angles.dtype)
    selected_complete = torch.zeros_like(accepted)
    min_gap = torch.full((len(rows),),torch.inf,device=fsm.device,dtype=angles.dtype)
    attempts = torch.zeros(len(rows),device=fsm.device,dtype=torch.int64)
    yaw_first_selected = torch.zeros_like(accepted)
    rejection = {key:torch.zeros_like(attempts) for key in (
        'duration','position','orientation','joint_margin','panel','initial_margin','initial_panel','retiming')}
    for _ in range((candidate_valid.shape[1]+7)//8):
        pending = torch.where(~accepted & available.any(-1))[0]
        if not len(pending):
            break
        order,eligibility,stops,completes = [],[],[],[]
        mask = available[pending].clone()
        for _ in range(8):
            priority = mask & preferred[pending]
            priority = torch.where(priority.any(-1)[:,None],priority,mask)
            selected,eligible,stop,complete = select_panel_arc(valid[pending],lengths[pending],
                angles[pending],priority,minimum_progress_rad=0.,prefer_longest=False)
            order.append(selected)
            eligibility.append(eligible)
            stops.append(stop)
            completes.append(complete)
            mask[~eligible] = False
            mask[torch.arange(len(pending),device=fsm.device),selected] = False
        available[pending] = mask
        order,eligible = torch.stack(order,1),torch.stack(eligibility,1)
        candidate_owner = pending[:,None].expand(-1,8).flatten()
        candidate_id = order.flatten()
        env_rows = rows[candidate_owner]
        start = state['ee_pos'][env_rows].double()
        start_q = state['ee_orn'][env_rows].double()
        target = bank['targets'][candidate_owner,candidate_id]
        orientation = bank['quats'][candidate_owner,candidate_id]
        normal = bank['normal'][candidate_owner]
        p,q,count = panel_arrival_path(start,start_q,target,orientation,normal,
            bank['jaw_local'][candidate_owner],capacity=capacity)
        arms = state['dof_pos'][env_rows][:,coordinator.arm_joint_ids].double()
        end_arms = bank['entry_joints'][candidate_owner,candidate_id]
        joint_route,joint_count = panel_joint_route(arms,end_arms,capacity,yaw_first=_yaw_first)
        tick = torch.arange(capacity,device=fsm.device)[None]
        roots = state['root_states'][env_rows].double()[:,None].expand(-1,capacity,-1).flatten(0,1)
        joint_p,joint_r,_ = b2z1_gripper_pose_and_jacobian(
            roots[:,:3],roots[:,[4,5,6,3]],joint_route.flatten(0,1))
        joint_p = joint_p.reshape(-1,capacity,3)
        joint_q = _matrix_quat(joint_r).reshape(-1,capacity,4)
        fraction = (tick/(count-1).clamp_min(1)[:,None]).clamp_max(1.)
        initial = arms[:,None]+fraction[...,None]*(end_arms-arms)[:,None]
        initial = torch.stack((joint_route,initial),1).flatten(0,1)
        p,q = torch.stack((joint_p,p),1).flatten(0,1),torch.stack((joint_q,q),1).flatten(0,1)
        count = torch.stack((joint_count,count),1).flatten()
        candidate_owner,candidate_id,env_rows,normal = (
            value.repeat_interleave(2,0) for value in (candidate_owner,candidate_id,env_rows,normal))
        eligible = eligible.repeat_interleave(2,1)
        feasible = eligible.flatten() & (count<=capacity-24)
        rejection['duration'].scatter_add_(0,candidate_owner,(eligible.flatten() & ~feasible).long())
        count = torch.where(feasible,count,torch.zeros_like(count))
        attempts.scatter_add_(0,candidate_owner,feasible.long())
        result,owner,sample = coordinator._inverse_pose_bank(fsm,state,env_rows,p,q,count,
            initial_arm_joint_positions=initial)
        if not len(owner):
            continue
        limits = coordinator.joint_limits.to(result.arm_joint_positions)
        margin = torch.minimum(result.arm_joint_positions-limits[:,0],
                               limits[:,1]-result.arm_joint_positions).amin(-1)
        wall_front = ((bank['tip_surface'][candidate_owner,candidate_id]
            - fsm.hinge_world[env_rows].double())*normal).sum(-1)
        gap = coordinator._panel_arrival_envelope.clearance(
            state['root_states'][env_rows[owner]],result.arm_joint_positions,
            state['gripper_position'][env_rows[owner],0],fsm.hinge_world[env_rows[owner]],normal[owner],
            bank['tangent'][candidate_owner[owner]],torch.ones_like(wall_front[owner]),
            wall_front[owner],panel_plane=True)
        good = (torch.isfinite(result.position_error).all(-1)
            & (torch.linalg.vector_norm(result.position_error,dim=-1)<=.001)
            & (torch.linalg.vector_norm(result.orientation_error,dim=-1)<=.005)
            & (margin>=.05) & (gap>=.003))
        checks = dict(position=torch.linalg.vector_norm(result.position_error,dim=-1)>.001,
            orientation=torch.linalg.vector_norm(result.orientation_error,dim=-1)>.005,
            joint_margin=margin<.05,panel=gap<.003,
            initial_margin=(sample==0)&(margin<.05),initial_panel=(sample==0)&(gap<.003))
        for key,failed in checks.items():
            per_path = torch.zeros(len(env_rows),device=fsm.device,dtype=torch.int64)
            per_path.scatter_reduce_(0,owner,failed.long(),reduce='amax')
            rejection[key].scatter_add_(0,candidate_owner,per_path)
        bad = torch.zeros(len(env_rows),device=fsm.device,dtype=torch.int64)
        bad.scatter_reduce_(0,owner,(~good).long(),reduce='amax')
        path_gap = torch.full((len(env_rows),),torch.inf,device=fsm.device,dtype=gap.dtype)
        path_gap.scatter_reduce_(0,owner,gap,reduce='amin')
        command_p,command_q = p.clone(),q.clone()
        joints = state['dof_pos'][env_rows][:,coordinator.arm_joint_ids].double()[:,None].expand(-1,capacity,-1).clone()
        command_p[owner,sample] = result.command_position_world.to(command_p.dtype)
        command_q[owner,sample] = result.command_orientation_world_wxyz.to(command_q.dtype)
        joints[owner,sample] = result.arm_joint_positions.to(joints.dtype)
        timed_p,timed_q,timed_count,overflow = retime_first_pull_paths(
            command_p,command_q,joints,count.clamp_min(1),max_joint_step=.02,
            max_position_step=.003,max_rotation_step=.025)
        feasible &= (bad==0) & ~overflow & (timed_count<=capacity-24)
        rejection['retiming'].scatter_add_(0,candidate_owner,
            (overflow | (timed_count>capacity-24)).long())
        feasible = feasible.reshape(-1,16)
        success = feasible.any(-1)
        slot = feasible.long().argmax(-1)
        selected_path = torch.arange(len(pending),device=fsm.device)[success]*16+slot[success]
        done = pending[success]
        accepted[done] = True
        yaw_first_selected[done] = _yaw_first
        chosen[done] = candidate_id[selected_path]
        selected_stop[done] = torch.stack(stops,1).repeat_interleave(2,1)[success,slot[success]]
        selected_complete[done] = torch.stack(completes,1).repeat_interleave(2,1)[success,slot[success]]
        min_gap[done] = path_gap[selected_path].to(min_gap.dtype)
        fsm.panel_arrival_path.install_tensor(rows[done],timed_p[selected_path].float(),
            timed_q[selected_path].float(),timed_count[selected_path])
    retry = torch.where(~accepted & has_arc.reshape_as(candidate_valid).any(-1))[0]
    if not _yaw_first and len(retry):
        retry_bank = {key:value[retry] for key,value in bank.items()}
        picked,passed,stop,complete = validate_panel_arrivals(
            coordinator,fsm,state,rows[retry],retry_bank,valid[retry],lengths[retry],angles[retry],
            candidate_valid[retry],_yaw_first=True)
        diagnostics = coordinator.path_diagnostics[-1][2]
        attempts[retry] += diagnostics['candidates_checked']
        for key in rejection:
            rejection[key][retry] += diagnostics['rejected_'+key]
        done = retry[passed]
        accepted[done] = True
        yaw_first_selected[done] = True
        chosen[done],selected_stop[done],selected_complete[done] = picked[passed],stop[passed],complete[passed]
        min_gap[done] = diagnostics['min_arm_plane_clearance_m'][passed]
    coordinator.path_diagnostics.append(('panel_arrival_path',rows.clone(),dict(
        accepted=accepted,candidates_checked=attempts,min_arm_plane_clearance_m=min_gap,
        standing_height_candidates=preferred.sum(-1),
        selected_standing_height_preferred=accepted & preferred.gather(1,chosen[:,None]).squeeze(1),
        yaw_first_fallback=yaw_first_selected,
        length=fsm.panel_arrival_path.length[rows].clone(),**{
            'rejected_'+key:value for key,value in rejection.items()})))
    return chosen,accepted,selected_stop,selected_complete
