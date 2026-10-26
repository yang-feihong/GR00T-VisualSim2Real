"""Compare recorded pull-door contact baselines without starting a simulator."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def intervals(mask):
    indices = np.flatnonzero(mask)
    return [part for part in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1) if len(part)]


def analyze(dataset):
    episodes = []
    for shard in sorted((dataset / 'shards').glob('episode-seed*')):
        seed = int(shard.name.split('seed')[-1])
        files = list(shard.glob('**/raw_audit/**/*.parquet'))
        if len(files) != 1:
            raise RuntimeError(f'{shard}: expected one raw audit, got {len(files)}')
        audit = pq.read_table(files[0]).to_pydict()
        stage = np.asarray(audit['autoopen_stage'])
        root = np.asarray(audit['root_position_world'])
        quat = np.asarray(audit['root_quaternion_xyzw'])
        x, y, z, w = quat.T
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        command = np.asarray(audit['base_command_dispatched'])
        moving = np.abs(command).max(-1) > 1.e-7
        force = np.zeros(len(stage))
        penetration = np.zeros(len(stage))
        for path in shard.glob('**/raw_contacts/**/*.parquet'):
            for contact in pq.read_table(path).to_pylist():
                a, b = contact['body_a_path'], contact['body_b_path']
                if '/Robot/' not in a:
                    a, b = b, a
                # Exclude handle and floor loads; retain all robot-panel contacts.
                if '/Robot/' not in a or 'panel' not in b.lower():
                    continue
                frame = contact['frame_index']
                if 0 <= frame < len(stage):
                    force[frame] = max(force[frame], contact['max_contact_force'])
                    penetration[frame] = max(penetration[frame], contact['max_penetration_depth'])
        stages = []
        for name in ('PULL_BASE_ADVANCE', 'CONTACT_PANEL', 'PUSH_OPEN', 'PULL_ARM_RETRACT', 'THROUGH'):
            for visit, rows in enumerate(intervals(stage == name), 1):
                delta = yaw[rows] - yaw[rows[0]]
                angle = np.arctan2(np.sin(delta), np.cos(delta))
                stationary = ~moving[rows]
                stages.append(dict(stage=name, visit=visit,
                    first_frame=int(rows[0]), last_frame=int(rows[-1]), frames=len(rows),
                    commanded_motion_frames=int(moving[rows].sum()),
                    command_restarts=int(np.sum(stationary[:-1] & ~stationary[1:])),
                    max_base_displacement_cm=float(np.linalg.norm(root[rows, :2] - root[rows[0], :2], axis=-1).max() * 100),
                    final_base_displacement_cm=float(np.linalg.norm(root[rows[-1], :2] - root[rows[0], :2]) * 100),
                    max_base_yaw_change_deg=float(np.degrees(np.abs(angle).max())),
                    min_base_height_m=float(root[rows, 2].min()),
                    peak_robot_panel_pair_force_n=float(force[rows].max()),
                    peak_robot_panel_penetration_mm=float(penetration[rows].max() * 1000),
                    door_angle_start_deg=float(np.degrees(audit['door_hinge_position'][rows[0]])),
                    door_angle_end_deg=float(np.degrees(audit['door_hinge_position'][rows[-1]]))))
        metadata = next(dataset.glob(f'logs/rank*-control/seed_{seed}/metadata.json'))
        meta = json.loads(metadata.read_text())
        acceptance = json.loads((shard / 'meta/acceptance_report.json').read_text())
        episodes.append(dict(seed=seed, control_completed=meta['success'],
            failure_reason=meta.get('failure_reason'), failure_stage=meta.get('failure_stage'),
            strict_accepted=acceptance.get('accepted_episodes') == 1,
            acceptance_report=acceptance, stages=stages,
            selected_contact=meta.get('acceptance_diagnostics', {}).get('panel_grid_inverse'),
            video=str(dataset / 'six_camera_mosaics' / f'seed{seed}_six_camera_mosaic.mp4')))
    return dict(dataset=str(dataset), workflow=json.loads((dataset / 'workflow.json').read_text()), episodes=episodes)


def compare_prefixes(datasets):
    comparisons = []
    for dataset in datasets[1:]:
        for seed in range(1700, 1704):
            audits = [pq.read_table(next((root / 'shards' / f'episode-seed{seed}').glob(
                '**/raw_audit/**/*.parquet'))).to_pydict() for root in (datasets[0], dataset)]
            starts = [next((i for i, s in enumerate(a['autoopen_stage']) if s == 'CONTACT_PANEL'),
                           len(a['autoopen_stage'])) for a in audits]
            length = min(starts)
            fields = {}
            for key in ('root_position_world', 'root_quaternion_xyzw', 'all_joint_position',
                        'door_hinge_position', 'base_command_dispatched', 'ee_inverse_input_world'):
                a, b = [np.asarray(a[key][:length]) for a in audits]
                fields[key] = dict(exact=bool(np.array_equal(a, b, equal_nan=True)),
                    max_abs_difference=float(np.nanmax(np.abs(a - b))) if length else None)
            comparisons.append(dict(dataset=str(dataset), reference=str(datasets[0]), seed=seed,
                compared_frames=length, contact_entry_frames=starts,
                stages_exact=audits[0]['autoopen_stage'][:length] == audits[1]['autoopen_stage'][:length],
                fields=fields))
    return comparisons


def verify_videos(dataset):
    import cv2

    videos = []
    for seed in range(1700, 1704):
        path = dataset / 'six_camera_mosaics' / f'seed{seed}_six_camera_mosaic.mp4'
        capture = cv2.VideoCapture(str(path))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if count <= 0 or abs(fps - 50.) > .01:
            raise RuntimeError(f'{path}: invalid frame count or rate')
        hashes, spreads = [], []
        tile_hashes = [[] for _ in range(6)]
        for index in (0, count // 2, count - 1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame.shape[:2] != (720, 1440):
                raise RuntimeError(f'{path}: unreadable frame {index} or unexpected mosaic dimensions')
            hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
            spreads.append(float(frame.std()))
            for tile in range(6):
                row, col = divmod(tile, 3)
                interior = frame[row*360+30:(row+1)*360-20, col*480+20:(col+1)*480-20]
                if interior.std() < 1.:
                    raise RuntimeError(f'{path}: blank camera tile {tile} at {index}')
                tile_hashes[tile].append(hashlib.sha256(interior.tobytes()).hexdigest())
        capture.release()
        if len(set(hashes)) < 2 or any(len(set(tile)) < 2 for tile in tile_hashes):
            raise RuntimeError(f'{path}: frozen mosaic or camera tile')
        videos.append(dict(seed=seed, path=str(path), frames=count, fps=fps,
            sample_hashes=hashes, frame_std=spreads, tile_hashes=tile_hashes))
    if len({v['sample_hashes'][0] for v in videos}) != 4:
        raise RuntimeError(f'{dataset}: identical first frames across seeds')
    return videos


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-videos', action='store_true')
    args = parser.parse_args()
    result = [analyze(path) for path in args.dataset]
    if args.verify_videos:
        for dataset, run in zip(args.dataset, result):
            run['video_checks'] = verify_videos(dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    if len(args.dataset) > 1:
        args.output.with_name(args.output.stem + '_prefixes.json').write_text(
            json.dumps(compare_prefixes(args.dataset), indent=2) + '\n')
    for run in result:
        print(run['workflow'].get('pull_panel_contact_baseline'), run['dataset'])
        for episode in run['episodes']:
            pushes = [s for s in episode['stages'] if s['stage'] == 'PUSH_OPEN']
            print(episode['seed'], episode['control_completed'], episode['failure_reason'],
                  [(s['frames'], round(s['max_base_displacement_cm'], 2), s['commanded_motion_frames'],
                    round(s['peak_robot_panel_pair_force_n'], 1)) for s in pushes])
