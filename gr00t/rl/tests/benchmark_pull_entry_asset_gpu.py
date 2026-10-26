"""Offline production USD geometry benchmark. Does not import/start a simulator.

Requires USD Python bindings and their shared libraries on the environment paths.
The pose is synthetic: this measures actual collider topology, not task success.
"""
import argparse
import time
from pathlib import Path

import torch

from gr00t.rl.auto_open.pull_entry_gpu import PullEntryGPU, _silhouette
from gr00t.rl.tests.test_auto_open_pull_entry_gpu import fixture, plan


def main():
    from pxr import Usd
    from gr00t.rl.auto_open.geometry import ROBOT_COLLIDER_SPECS, _read_collider_points

    parser = argparse.ArgumentParser()
    parser.add_argument('--environments', type=int, default=4)
    parser.add_argument('--candidates', type=int, default=8)
    parser.add_argument('--trials', type=int, default=1)
    args = parser.parse_args()
    asset = Path(__file__).resolve().parents[1]/'data/robots/b2z1_lab30/b2z1.usd'
    stage = Usd.Stage.Open(str(asset))
    geometries = {}
    for body_name, key, tokens in ROBOT_COLLIDER_SPECS:
        if key == 'tcp':
            continue
        body = next(p for p in Usd.PrimRange(stage.GetDefaultPrim(), Usd.TraverseInstanceProxies())
                    if p.GetName().lower() == body_name.lower())
        geometries[key] = _read_collider_points(stage, body, tokens)
    _, state, controller = fixture(device='cuda', environments=args.environments)
    controller.center_geometry = geometries
    cache = PullEntryGPU.from_controllers([controller]*args.environments, device='cuda', max_samples=1024)
    print('ASSET', asset, 'parts', len(cache.parts),
          'vertices', sum(x['original_vertices'] for x in cache.collider_metadata),
          'hull_vertices', sum(x['hull_vertices'] for x in cache.collider_metadata),
          'edges', sum(x['edges'] for x in cache.collider_metadata), flush=True)
    print('PROJECTED_COUNTS', [_silhouette(p.points_body[:1],p)[2].sum().item() for p in cache.parts], flush=True)
    for trial in range(args.trials):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = plan(cache, state, args.candidates)
        torch.cuda.synchronize()
        print('RESULT', dict(trial=trial, environments=args.environments, candidates=args.candidates,
            seconds=time.perf_counter()-start, peak_MB=torch.cuda.max_memory_allocated()/1e6,
            feasible=result['feasible'].sum().item(), geometry_safe=result['geometry_safe'].sum().item(),
            route_safe=result['route_safe'].sum().item(), max_length=result['lengths'].max().item()), flush=True)
        del result


if __name__ == '__main__':
    main()
