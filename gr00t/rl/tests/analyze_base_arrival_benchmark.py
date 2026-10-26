"""Generate a reproducible report and audit the station benchmark traces."""

import argparse
import json
from pathlib import Path

import torch


def audit_run(directory):
    result = json.loads((directory / 'summary.json').read_text())
    checked = 0
    for path in sorted(directory.glob('rank[0-9].json')):
        rows = json.loads(path.read_text())['trials']
        saved = torch.load(path.with_name(path.stem + '_trace.pt'), map_location='cpu', weights_only=True)
        trace, columns = saved['trace'], saved['columns']
        assert torch.isfinite(trace).all(), path
        command = [columns.index(c) for c in ('cmd_vx', 'cmd_vy', 'cmd_wz')]
        effective = [columns.index(c) for c in ('effective_vx', 'effective_vy', 'effective_wz')] if 'effective_vx' in columns else command
        for i, row in enumerate(rows):
            if not row['success']:
                continue
            step = round(row['elapsed_s'] / saved['dt'])
            tail = trace[step - 100:step, i]
            assert len(tail) == 100
            assert (tail[:, command] == 0).all(), (path, i, 'nonzero stopped command')
            assert (tail[:, effective] == 0).all(), (path, i, 'nonzero effective command')
            assert (tail[:, columns.index('distance')] < row['profile']['outer_m']).all()
            assert (tail[:, columns.index('yaw_error')] < row['profile']['outer_yaw']).all()
            assert (torch.linalg.vector_norm(tail[:, [columns.index(c) for c in ('vx','vy','vz')]], dim=-1) < .02).all()
            if 'wx' in columns:
                assert (torch.linalg.vector_norm(tail[:, [columns.index(c) for c in ('wx','wy','wz')]], dim=-1) < .05).all()
            checked += 1
    return result, checked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    datasets = []
    audits = []
    for path in args.runs:
        data, count = audit_run(path)
        datasets.append((path, data))
        audits.append(dict(run=str(path), stopped_trajectories_checked=count))
    lines = ['# Base Arrival Controller Measurement', '',
        f'Total: {sum(len(d["trials"]) for _, d in datasets)} goal/posture/profile trials; '
        f'{sum(d["elapsed_s"] for _, d in datasets)/60:.2f} min summed sweep runtime (excluding development and smoke checks).', '',
        '## Interpretation', '',
        '- Practical candidate: position inner 1-1.5 cm, outer 8 cm; yaw inner 0.015 rad, outer 0.10 rad.',
        '- 1 cm prioritizes accuracy; 1.5 cm leaves more room for cases that cannot enter the 1 cm braking region.',
        '- An inner threshold is a braking trigger, not a final-error guarantee. The outer threshold remains the allowed error bound.',
        '- Very small inner thresholds (0.25-0.5 cm) improve typical error but introduce near-target timeouts.',
        '- No AutoOpen production defaults were changed by this experiment.', '',
        '## Scope', '',
        '- Production `_track_station`, arrival latch, stability criterion, low-level checkpoint and 16/2 physics.',
        '- Flat floor, no door/wall contact. Same goal scenarios paired across threshold profiles.',
        '- Held middle and restored arm postures. No camera or data-collection overhead.',
        '- 25-frame zero-command/stability gate, followed by 100 further consecutive confirmed frames (2 s).',
        '- One pass means one braking window, no motion resumption, no safety failure and no timeout (30 s).',
        '- Accuracy counts below use all attempted trials as denominator, including failures.',
        '- Structured scenario coverage, not independent random trials or a universal accuracy guarantee.', '',
        '## Results', '']
    for path, data in datasets:
        lines.extend([f'### {data["phase"]}', '', f'Run: `{path}`; wall time {data["elapsed_s"]:.1f} s.', '',
            '| Inner/Outer cm | Yaw Inner/Outer rad | Stopped | One Pass | <=3cm/2deg One Pass | <=5cm/3deg One Pass | Position Median/P90 cm | Yaw Median/P90 deg | Median Time s |',
            '|---|---|---|---|---|---|---|---|---|'])
        for r in data['profiles']:
            q, n = r['profile'], r['trials']
            def number(value, scale=1):
                return '-' if value is None else f'{value * scale:.2f}'
            lines.append(f"| {q['inner_m']*100:g}/{q['outer_m']*100:g} | {q['inner_yaw']:g}/{q['outer_yaw']:g} "
                f"| {r['success_count']}/{n} | {r['one_pass_count']}/{n} | {r['within_3cm_2deg_count']}/{n} "
                f"| {r['within_5cm_3deg_count']}/{n} | {number(r['distance_median_m'],100)}/{number(r['distance_p90_m'],100)} "
                f"| {number(r['yaw_median_deg'])}/{number(r['yaw_p90_deg'])} | {number(r['elapsed_median_s'])} |")
        lines.append('')
        for r in data['profiles']:
            failed = {k: v for k, v in r['failures'].items() if v}
            if failed:
                lines.append(f"- Profile {r['profile_index']}: {failed}; mean restarts {r['mean_restarts']:.2f}.")
        lines.append('')
    lines.extend(['## Trace Audit', '',
        'Every stopped trajectory was checked for 100 consecutive zero-command frames, real velocity <0.02 m/s, '
        'position/yaw inside its outer bounds, and finite trace values. Effective commands are also audited when recorded.', '',
        *[f"- `{r['run']}`: {r['stopped_trajectories_checked']} stopped trajectories checked." for r in audits], '',
        '## Reuse', '',
        'Run inside the existing doorman_train container, with the repository and low-level library on PYTHONPATH:', '',
        '```bash',
        'python gr00t/rl/tests/benchmark_base_arrival.py --output outputs/base_arrival/NEW_RUN --gpus 0 1 2 3 --phase position',
        'python gr00t/rl/tests/benchmark_base_arrival.py --output outputs/base_arrival/NEW_YAW_RUN --gpus 0 1 2 3 --phase yaw --position .01 .08 --expanded-cases',
        'python gr00t/rl/tests/benchmark_base_arrival.py --output outputs/base_arrival/NEW_VALIDATION --gpus 0 1 2 3 --phase validation --yaw .015 .10 --expanded-cases',
        '```', ''])
    (args.output / 'report.md').write_text('\n'.join(lines))
    (args.output / 'trace_audit.json').write_text(json.dumps(audits, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, key, label in zip(axes, ('distance_p90_m', 'yaw_p90_deg'), ('Position P90 (cm)', 'Yaw P90 (deg)')):
        for path, data in datasets:
            rows = [r for r in data['profiles'] if r[key] is not None]
            x = [r[key] * (100 if key.endswith('_m') else 1) for r in rows]
            y = [r['one_pass_rate'] * 100 for r in rows]
            ax.scatter(x, y, label=data['phase'], alpha=.8)
            if data['phase'] == 'validation':
                for a, b, r in zip(x, y, rows):
                    if r['profile_index'] not in (0, 5, 10):
                        continue
                    q = r['profile']
                    ax.annotate(f"{q['inner_m']*100:g}/{q['outer_m']*100:g} cm", (a,b),
                        fontsize=8, xytext=(-45,-30), textcoords='offset points',
                        arrowprops=dict(arrowstyle='-', lw=.7))
        ax.set(xlabel=label, ylabel='One-pass completion (%)', ylim=(0, 103))
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(args.output / 'accuracy_vs_one_pass.png', dpi=160)
    plt.close(fig)
    print(f'Report: {args.output / "report.md"}; audited {sum(r["stopped_trajectories_checked"] for r in audits)} stopped trajectories')


if __name__ == '__main__':
    main()
