"""Offline common feasible XY/yaw stations across recorded pull-door seeds."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from pxr import Usd
from scipy.spatial.transform import Rotation

from probe_auto_open_pull_clearance import recorded_collision_clouds, rotation
from probe_auto_open_pull_stations import load_records, evaluate_stations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[1700, 1701, 1702, 1703])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--x-range', nargs=3, type=float, default=[.30, .70, .025])
    parser.add_argument('--y-range', nargs=3, type=float, default=[.35, .90, .025])
    parser.add_argument('--yaw-range', nargs=3, type=float, default=[-140., -85., 5.])
    parser.add_argument('--clearance-mode', choices=('intersection', 'union'), default='intersection')
    args = parser.parse_args()
    args.quiet = True
    start = time.perf_counter()
    records = load_records(args)
    def values(bounds):
        return np.arange(bounds[0], bounds[1] + bounds[2] * .1, bounds[2])
    grid = np.array(np.meshgrid(values(args.x_range), values(args.y_range), values(args.yaw_range), indexing='ij')).reshape(3, -1).T
    grid_t = torch.tensor(grid, dtype=torch.float32, device=args.device)
    positions, headings, margins = [], [], []
    for record in records:
        root = record['root']
        clouds, anchor_error = recorded_collision_clouds(Usd.Stage.Open(record['asset_path']), np.asarray(record['body_poses']))
        points = np.concatenate([points for _, points in clouds])
        old_yaw = rotation(root).as_euler('xyz')[2]
        level_local = Rotation.from_euler('z', -old_yaw).apply(points - root[:3])
        local = torch.tensor(level_local[:, :2], dtype=torch.float32, device=args.device)
        normal = torch.tensor(record['normal'][:2], dtype=torch.float32, device=args.device)
        tangent = torch.tensor(record['tangent'][:2], dtype=torch.float32, device=args.device)
        hinge = torch.tensor(record['hinge'][:2], dtype=torch.float32, device=args.device)
        origin = hinge + record['wall'] * normal + record['frame'] * tangent
        xy = origin + grid_t[:, :1] * normal + grid_t[:, 1:2] * tangent
        forward = torch.cos(torch.deg2rad(grid_t[:, 2:3])) * normal + torch.sin(torch.deg2rad(grid_t[:, 2:3])) * tangent
        yaw = torch.atan2(forward[:, 1], forward[:, 0])
        c, s = yaw.cos()[:, None], yaw.sin()[:, None]
        offsets = torch.stack((c * local[:, 0] - s * local[:, 1], s * local[:, 0] + c * local[:, 1]), -1)
        moved = xy[:, None] + offsets - origin
        wall = (moved * normal).sum(-1).amin(-1)
        frame = (moved * tangent).sum(-1).amin(-1)
        positions.append(xy)
        headings.append(yaw)
        margins.append(torch.stack((wall, frame), -1))
        print(f"Seed {record['seed']}: verified body mapping, max anchor error {anchor_error * 1000:.6f} mm", flush=True)
    positions, headings, margins = torch.stack(positions), torch.stack(headings), torch.stack(margins)
    per_seed_clearance = (margins[..., 0] >= .07) & (margins[..., 1] >= .05)
    clearance_pass = (per_seed_clearance.all(0) if args.clearance_mode == 'intersection'
                      else per_seed_clearance.any(0))
    selected = torch.where(clearance_pass)[0]
    print(f'{len(grid)} station combinations; {len(selected)} pass whole-robot clearance for all seeds', flush=True)
    output = dict(dataset=str(args.dataset), seeds=args.seeds,
        coordinate_definition=dict(origin='Free frame boundary intersected with robot-facing wall plane',
            x='Outward wall normal, meters', y='Along wall away from opening, meters',
            yaw='Degrees from local +x toward local +y; -90 points toward hinge along wall; mirrored per door'),
        assumptions='Recorded retracted posture/height/tilt. Static whole-robot collision envelopes; production first-insertion path and inverse. No simulation, walking-path or later-pry verification.',
        sampled_count=len(grid), common_clearance_count=len(selected),
        ranges=dict(x=args.x_range, y=args.y_range, yaw=args.yaw_range),
        thresholds=dict(wall_m=.07, frame_m=.05, inverse_position_m=.004, inverse_orientation_deg=1.),
        common_pass=[], evaluated=[])
    for begin in range(0, len(selected), args.batch_size):
        ids = selected[begin:begin + args.batch_size]
        labels = torch.stack((torch.arange(len(ids), device=args.device), torch.zeros(len(ids), device=args.device)), -1)
        result = evaluate_stations(args, records, positions[:, ids], headings[:, ids], labels)
        rows = result['results']
        for j, index in enumerate(ids.cpu().tolist()):
            seed_rows = [r for r in rows if r['position_angle_deg'] == j]
            item = dict(x_m=float(grid[index, 0]), y_m=float(grid[index, 1]), yaw_deg=float(grid[index, 2]),
                min_wall_mm=float(margins[:, index, 0].min()) * 1000,
                min_frame_mm=float(margins[:, index, 1].min()) * 1000,
                per_seed=[dict(seed=r['seed'], clearance_pass=bool(per_seed_clearance[k, index]),
                    whole_path_pass=r['whole_path_pass'] if bool(per_seed_clearance[k, index]) else 0,
                    selected_height_m=r['selected_height_m'], selected_insertion_m=r['selected_insertion_m'],
                    min_max_position_error_mm=r['min_max_position_error_mm']) for k, r in enumerate(seed_rows)])
            if len(seed_rows) != len(records):
                raise ValueError('Missing seed evaluation')
            output['evaluated'].append(item)
            if all(r['whole_path_pass'] > 0 for r in item['per_seed']):
                output['common_pass'].append(item)
        print(f"Evaluated {min(begin + len(ids), len(selected))}/{len(selected)}; common inverse passes {len(output['common_pass'])}; elapsed {time.perf_counter() - start:.1f}s", flush=True)
        output['wall_time_s'] = time.perf_counter() - start
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + '\n')
    output['wall_time_s'] = time.perf_counter() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(f"Report: {args.output}; {len(output['common_pass'])} common passes; {output['wall_time_s']:.2f}s", flush=True)


if __name__ == '__main__':
    main()
