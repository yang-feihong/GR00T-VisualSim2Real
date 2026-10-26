"""Offline comparison of station regions using production GPU path validation."""

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time
from types import MethodType, SimpleNamespace

import numpy as np
import torch
from pxr import Usd
from scipy.spatial.transform import Rotation

from probe_auto_open_pull_stations import load_records
from probe_auto_open_pull_clearance import recorded_collision_clouds, rotation
from gr00t.rl.auto_open.first_pull_station import station_candidates, first_insertion_residuals
from gr00t.rl.auto_open.panel_motion import PullPanelConfig
from gr00t.rl.auto_open.pull_entry_gpu import PullEntryGPU
from gr00t.rl.auto_open.pull_station import PullStationConfig
from gr00t.rl.auto_open.tensor_fsm import _pull_cycle_uniform


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--failure-reason', default='pull_initial_station_no_clearance_and_full_insertion_feasible_candidate')
    parser.add_argument('--xy-error-m', type=float, default=0.)
    parser.add_argument('--yaw-error-deg', type=float, default=0.)
    parser.add_argument('--region', action='append', nargs=7,
                        metavar=('NAME', 'XMIN', 'XMAX', 'YMIN', 'YMAX', 'YAWMIN', 'YAWMAX'))
    args = parser.parse_args()
    failures = sorted(f['seed'] for p in args.dataset.glob('logs/rank*-control/summary.json')
                      for f in json.loads(p.read_text())['failures']
                      if f['reason'] == args.failure_reason)
    if args.seeds is None:
        if not failures or args.count < 4:
            parser.error('Need at least four samples and matching failures, or explicit --seeds')
        selected = np.linspace(0, len(failures) - 1, args.count - 4).astype(int)
        args.seeds = sorted(set([1700, 1701, 1702, 1703] + [failures[i] for i in selected]))
    start = time.perf_counter()
    records = load_records(args)
    def tensor(value):
        return torch.as_tensor(np.asarray(value), dtype=torch.float32, device=args.device)
    def values(key):
        return tensor([r[key] for r in records])
    roots, arms = values('root'), values('arm')
    state = dict(root_states=roots, dof_pos=arms, door_joint_position=values('door_angle')[:, None])
    cache = SimpleNamespace(max_samples=1024)
    cache._sample = MethodType(PullEntryGPU._sample, cache)
    for target, key in (('hinge_world', 'hinge'), ('door_normal', 'normal'), ('tangent', 'tangent'),
                        ('panel_width', 'planner_width'), ('panel_center', 'center'),
                        ('panel_half_thickness', 'half'), ('hinge_axis_sign', 'sign')):
        setattr(cache, target, values(key))
    cache.hinge_idx = torch.zeros(len(records), dtype=torch.int64, device=args.device)
    stage = Usd.Stage.Open(records[0]['asset_path'])
    joints = {p.GetName(): p for p in stage.Traverse() if p.GetName().startswith('joint')}
    limits = tensor([[math.radians(joints[f'joint{i}'].GetAttribute(f'physics:{name}Limit').Get())
                      for name in ('lower', 'upper')] for i in range(1, 7)])
    clouds = []
    anchor_error = 0.
    for record in records:
        cloud, error = recorded_collision_clouds(stage, record['body_poses'])
        points = np.concatenate([p for _, p in cloud])
        yaw = rotation(record['root']).as_euler('xyz')[2]
        clouds.append(Rotation.from_euler('z', -yaw).apply(points - record['root'][:3])[:, :2])
        anchor_error = max(anchor_error, error)
    local = tensor(np.stack(clouds))
    config = PullPanelConfig()
    pose_grid = torch.cartesian_prod(tensor(config.height_offsets_m), tensor(config.insertion_lengths_m))
    pose_grid[:, 0] += config.height_nominal_m
    seeds = torch.tensor(args.seeds, dtype=torch.int64, device=args.device)
    uniform = _pull_cycle_uniform(seeds, torch.full_like(seeds, 1 + 0x53544154), 32)[..., :3]
    regions = dict(old=PullStationConfig(),
        expanded=replace(PullStationConfig(), first_x_range_m=(.45, .55), first_y_range_m=(.70, .95),
                         first_yaw_range_deg=(-130., -105.)),
        central=replace(PullStationConfig(), first_x_range_m=(.475, .525), first_y_range_m=(.75, .90),
                        first_yaw_range_deg=(-125., -115.)))
    if args.region:
        regions = {}
        for name, *bounds in args.region:
            x0, x1, y0, y1, a0, a1 = map(float, bounds)
            if not (x0 <= x1 and y0 <= y1 and a0 <= a1):
                parser.error('Region lower bounds must not exceed upper bounds')
            regions[name] = replace(PullStationConfig(), first_x_range_m=(x0, x1),
                first_y_range_m=(y0, y1), first_yaw_range_deg=(a0, a1))
    output = dict(dataset=str(args.dataset), seeds=args.seeds, max_anchor_error_m=anchor_error,
                  assumptions='Recorded station-entry base height, tilt and retracted joints; static envelopes; no walking simulation.',
                  insertion_max_horizontal_radius_m=config.insertion_max_horizontal_radius_m,
                  regions={})
    for name, region in regions.items():
        tick = time.perf_counter()
        xy, yaw, samples = station_candidates(values('hinge'), values('normal'), values('tangent')[:, :2],
            values('frame'), values('wall'), uniform, region)
        perturbations = 1
        if args.xy_error_m or args.yaw_error_deg:
            delta = torch.cartesian_prod(
                tensor([-args.xy_error_m, 0., args.xy_error_m]),
                tensor([-args.xy_error_m, 0., args.xy_error_m]),
                tensor([-args.yaw_error_deg, 0., args.yaw_error_deg])).unique(dim=0)
            perturbations = len(delta)
            shift = (delta[:, :1] * values('normal')[:, None, :2]
                     + delta[:, 1:2] * values('tangent')[:, None, :2])
            xy = (xy[:, :, None] + shift[:, None]).flatten(1, 2)
            yaw = (yaw[:, :, None] + torch.deg2rad(delta[:, 2])[None, None]).flatten(1, 2)
        c, s = yaw.cos()[..., None], yaw.sin()[..., None]
        x, y = local[:, None, :, 0], local[:, None, :, 1]
        moved = xy[:, :, None] + torch.stack((c*x - s*y, s*x + c*y), -1) - values('hinge')[:, None, None, :2]
        wall = (moved * values('normal')[:, None, None, :2]).sum(-1).amin(-1) - values('wall')[:, None]
        frame = (moved * values('tangent')[:, None, None, :2]).sum(-1).amin(-1) - values('frame')[:, None]
        safe = (wall >= .07) & (frame >= .05)
        feasible, error, angle = first_insertion_residuals(cache, state,
            torch.arange(len(records), device=args.device), xy, yaw, safe, pose_grid,
            torch.arange(6, device=args.device), limits, values('frame'), values('wall'), values('hardware'), config)
        accepted = feasible.any(-1) & safe
        accepted = accepted.reshape(len(records), 32, perturbations).all(-1)
        output['regions'][name] = dict(x=region.first_x_range_m, y=region.first_y_range_m,
            yaw=region.first_yaw_range_deg, solved=int(accepted.any(-1).sum()),
            xy_error_m=args.xy_error_m, yaw_error_deg=args.yaw_error_deg,
            perturbations_per_station=perturbations,
            compute_s=time.perf_counter() - tick,
            per_seed=[dict(seed=r['seed'], clearance_candidates=int(safe[i].sum()),
                feasible_stations=int(accepted[i].sum()),
                best_position_error_m=float(error[i].min()) if torch.isfinite(error[i]).any() else None)
                for i, r in enumerate(records)])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + '\n')
        print(json.dumps(dict(region=name, solved=output['regions'][name]['solved'],
                              total=len(records), compute_s=output['regions'][name]['compute_s'])), flush=True)
    output['wall_time_s'] = time.perf_counter() - start
    args.output.write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
