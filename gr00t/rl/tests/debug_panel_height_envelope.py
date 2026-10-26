"""Offline endpoint robustness probe using the controller's standing height band."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from gr00t.rl.auto_open.panel_push_forecast import panel_candidate_bank
from gr00t.rl.scripts.probe_auto_open_arm_wall import Robot
from legged_gym.utils.ee_steady_inverse_gpu import batched_steady_ee_inverse_no_gravity


def endpoint_height_mask(bank, roots, joints, limits, heights):
    n,c = bank['targets'].shape[:2]
    h = len(heights)
    entry_base = roots[:,None].expand(n,c,-1).flatten(0,1)
    entry = batched_steady_ee_inverse_no_gravity(
        bank['targets'].flatten(0,1),bank['quats'].flatten(0,1),
        base_position_world=entry_base[:,:3],base_orientation_world_xyzw=entry_base[:,[4,5,6,3]],
        initial_arm_joint_positions=joints.flatten(0,1),joint_limits=limits)
    joints = entry.arm_joint_positions.reshape(n,c,6)
    base = roots[:,None,None].expand(n,c,h,-1).clone()
    base[:,:,:,2] = heights
    result = batched_steady_ee_inverse_no_gravity(
        bank['targets'][:,:,None].expand(n,c,h,3).flatten(0,2),
        bank['quats'][:,:,None].expand(n,c,h,4).flatten(0,2),
        base_position_world=base[:,:,:,:3].flatten(0,2),
        base_orientation_world_xyzw=base[:,:,:,[4,5,6,3]].flatten(0,2),
        initial_arm_joint_positions=joints[:,:,None].expand(n,c,h,6).flatten(0,2),
        joint_limits=limits)
    margin = torch.minimum(result.arm_joint_positions-limits[:,0],limits[:,1]-result.arm_joint_positions).amin(-1)
    passed = (result.position_error.norm(dim=-1)<=.001)&(result.orientation_error.norm(dim=-1)<=.005)&(margin>=.05)
    return passed.reshape(n,c,h)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('dataset',type=Path)
    parser.add_argument('inputs',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--min-height',type=float,default=.40)
    parser.add_argument('--max-height',type=float,default=.70)
    parser.add_argument('--samples',type=int,default=7)
    parser.add_argument('--height-shift',type=float,default=0.)
    parser.add_argument('--device',default='cuda:0')
    args = parser.parse_args()
    records = [r for path in args.inputs for r in json.loads(path.read_text())['records']]
    prov = json.loads((args.dataset/'shards'/f"episode-seed{records[0]['seed']}"/'meta/simulation_provenance.json').read_text())
    t = lambda x:torch.as_tensor(np.asarray(x),device=args.device,dtype=torch.float64)
    limits = t(Robot(prov['robot_asset']['path']).limits)
    heights = torch.linspace(args.min_height,args.max_height,args.samples,device=args.device,dtype=torch.float64)
    outputs = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    for begin in range(0,len(records),16):
        chunk = records[begin:begin+16]
        values = lambda key:t([r[key] for r in chunk])
        scalar = lambda x:torch.full((len(chunk),),x,device=args.device,dtype=torch.float64)
        roots = values('root')
        desired = values('desired').clone()
        desired[:,2] += args.height_shift
        bank = panel_candidate_bank(desired,values('orientation'),values('normal'),values('tangent'),
            values('jaw_local'),values('depth'),values('hinge'),roots,t([[.2,0.,.7]]).expand(len(chunk),-1),
            scalar(.03),values('edge'),values('bottom'),values('top'),scalar(.05),scalar(.7),
            values('handle_bounds'),values('sign'))
        n,c = bank['targets'].shape[:2]
        joints = values('arm')[:,None].expand(n,c,-1)
        per_height = endpoint_height_mask(bank,roots,joints,limits,heights)
        robust = per_height.all(-1)
        allowed = bank['geometry_valid']
        for i,r in enumerate(chunk):
            good = torch.where(robust[i]&allowed[i])[0]
            outputs.append(dict(seed=r['seed'],candidates=int(len(good)),
                per_height_count=(per_height[i]&allowed[i,:,None]).sum(0).tolist(),
                example_parameters=None if not len(good) else bank['parameters'][i,good[0]].tolist()))
        print('processed',min(begin+16,len(records)),flush=True)
    torch.cuda.synchronize()
    summary = dict(records=len(outputs),heights_m=heights.tolist(),candidate_height_shift_m=args.height_shift,compute_s=time.perf_counter()-started,
        with_candidate=sum(r['candidates']>0 for r in outputs),
        rejected_seeds=[r['seed'] for r in outputs if not r['candidates']])
    args.output.write_text(json.dumps(dict(summary=summary,records=outputs),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
