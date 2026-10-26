"""Offline endpoint inverse at entry and terminal base poses; no simulator."""
import numpy as np
import pyarrow.parquet as pq
import torch

from legged_gym.utils.ee_steady_inverse_gpu import batched_steady_ee_inverse_no_gravity
from gr00t.rl.auto_open.tensor_stage_planning import align_vectors, _quat_rotate, _quat_multiply, _yaw_quat
from legged_gym.utils.ee_steady_inverse_gpu import _cpu_orientation_error
from legged_gym.utils.constrained_ik import batched_constrained_dls_delta
from gr00t.rl.envs.door.b2z1_reset_geometry import b2z1_gripper_pose_and_jacobian


def audit_terminal(dataset, record, limits, device, *, use_replayed_endpoint=False):
    shard = dataset/'shards'/f"episode-seed{record['seed']}"
    raw = pq.read_table(next(shard.rglob('raw_audit/**/*.parquet'))).to_pydict()
    ids = np.flatnonzero(np.asarray(raw['autoopen_stage'])=='CONTACT_PANEL')
    ids = np.split(ids,np.flatnonzero(np.diff(ids)>1)+1)[-1]
    end = int(ids[-1])
    tensor = lambda value:torch.as_tensor(np.asarray(value),device=device,dtype=torch.float64)
    root = np.r_[raw['root_position_world'][end],np.asarray(raw['root_quaternion_xyzw'][end])[[3,0,1,2]]]
    target = tensor(raw['ee_inverse_input_world'][end])
    aligned = align_vectors(_quat_rotate(tensor([record['orientation']]),tensor([record['depth']])),
        -tensor([record['normal']]),tensor([record['orientation']]))
    offset = record['recorded']['selected_offset']
    orientation = _quat_multiply(_yaw_quat(tensor([record['sign']*np.deg2rad(offset[2])])),aligned)[0]
    if use_replayed_endpoint:
        target = tensor(record['arrival_endpoint_world'])
        orientation = tensor(record['arrival_orientation_world_wxyz'])
    # Reproduce both the locally occupied branch and alternate joint seeds.
    samples = torch.quasirandom.SobolEngine(6,scramble=False).draw(128).to(device=device,dtype=torch.float64)
    seeds = limits[:,0]+.05+samples*(limits[:,1]-limits[:,0]-.1)
    seeds = torch.cat((tensor(raw['arm_joint_position'][end])[None],tensor(record['arm'])[None],seeds))
    roots = tensor([record['root'],root])
    n = len(seeds)
    bases = roots[:,None].expand(-1,n,-1).flatten(0,1)
    result = batched_steady_ee_inverse_no_gravity(
        target[None].expand(2*n,-1),orientation[None].expand(2*n,-1),
        base_position_world=bases[:,:3],base_orientation_world_xyzw=bases[:,[4,5,6,3]],
        initial_arm_joint_positions=seeds.repeat(2,1),joint_limits=limits)
    position = result.position_error.norm(dim=-1).reshape(2,n)
    rotation = result.orientation_error.norm(dim=-1).reshape(2,n)
    exact = (position<=.001)&(rotation<=.005)
    output = dict(frame=end,counterfactual_endpoint=use_replayed_endpoint,
        evaluated_target_world=target.tolist(),published_target_world=raw['ee_inverse_input_world'][end],
        reconstructed_command_error_m=float((tensor(record['arrival_endpoint_world'])-target).norm()),
        logged_output_minus_input_m=(tensor(raw['ee_inverse_output_world'][end])-tensor(raw['ee_inverse_input_world'][end])).tolist(),
        actual_minus_input_m=(tensor(raw['ee_pose_world_xyzw'][end][:3])-tensor(raw['ee_inverse_input_world'][end])).tolist(),
        entry_root=record['root'],terminal_root=root.tolist(),branches={})
    for i,label in enumerate(('entry_base','terminal_base')):
        best = int((position[i]+rotation[i]).argmin())
        output['branches'][label] = dict(exact_seed_count=int(exact[i].sum()),
            current_branch_position_error_m=float(position[i,0]),
            current_branch_orientation_error=float(rotation[i,0]),
            entry_branch_position_error_m=float(position[i,1]),
            best_position_error_m=float(position[i,best]),best_orientation_error=float(rotation[i,best]),
            best_joint_solution=result.arm_joint_positions.reshape(2,n,6)[i,best].tolist())
    relaxed = limits.clone()
    relaxed[3,0] = -torch.pi
    result = batched_steady_ee_inverse_no_gravity(
        target[None].expand(n,-1),orientation[None].expand(n,-1),
        base_position_world=tensor(root[:3])[None].expand(n,-1),
        base_orientation_world_xyzw=tensor(root[[4,5,6,3]])[None].expand(n,-1),
        initial_arm_joint_positions=seeds,joint_limits=relaxed)
    exact = (result.position_error.norm(dim=-1)<=.001)&(result.orientation_error.norm(dim=-1)<=.005)
    output['relaxed_q4_lower_bound_only'] = dict(exact_seed_count=int(exact.sum()),
        exact_q4_range_rad=([float(result.arm_joint_positions[exact,3].min()),
            float(result.arm_joint_positions[exact,3].max())] if bool(exact.any()) else None),
        original_q4_lower_rad=float(limits[3,0]))
    controls = tensor([record['root'],root])
    controls[0,2] = float(root[2])
    controls[1,2] = float(record['root'][2])
    bases = controls[:,None].expand(-1,n,-1).flatten(0,1)
    result = batched_steady_ee_inverse_no_gravity(
        target[None].expand(2*n,-1),orientation[None].expand(2*n,-1),
        base_position_world=bases[:,:3],base_orientation_world_xyzw=bases[:,[4,5,6,3]],
        initial_arm_joint_positions=seeds.repeat(2,1),joint_limits=limits)
    pos,rot = result.position_error.norm(dim=-1).reshape(2,n),result.orientation_error.norm(dim=-1).reshape(2,n)
    output['height_only_counterfactual'] = {
        label:dict(exact_seed_count=int(((pos[i]<=.001)&(rot[i]<=.005)).sum()),
            best_position_error_m=float(pos[i,(pos[i]+rot[i]).argmin()]),
            best_orientation_error=float(rot[i,(pos[i]+rot[i]).argmin()]))
        for i,label in enumerate(('entry_pose_with_terminal_height','terminal_pose_with_entry_height'))}
    arm = tensor(raw['arm_joint_position'][end])[None].expand(2,-1).clone()
    base = tensor(root)[None].expand(2,-1)
    gains = tensor([1.,4.])[:,None]
    for _ in range(160):
        p,r,jacobian = b2z1_gripper_pose_and_jacobian(base[:,:3],base[:,[4,5,6,3]],arm)
        residual = torch.cat((target-p,_cpu_orientation_error(orientation[None].expand(2,-1),r)*gains),-1)
        delta = batched_constrained_dls_delta(jacobian,residual,arm,limits,.05)
        arm = arm+delta
    p,r,jacobian = b2z1_gripper_pose_and_jacobian(base[:,:3],base[:,[4,5,6,3]],arm)
    output['online_math_fixed_base'] = {
        f'orientation_gain_{gain}':dict(output_minus_input_m=(p[i]-target).tolist(),
            orientation_error=float(_cpu_orientation_error(orientation[None],r[i:i+1]).norm()),
            final_joint_delta_rad=delta[i].tolist(),arm_joint_positions=arm[i].tolist())
        for i,gain in enumerate((1,4))}
    return output
