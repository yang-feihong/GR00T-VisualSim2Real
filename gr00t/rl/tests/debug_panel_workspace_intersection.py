"""Offline continuous radial-interval check; no simulator or control changes."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from gr00t.rl.auto_open import panel_push_forecast as forecast
from gr00t.rl.auto_open.tensor_stage_planning import _quat_rotate, _yaw_quat


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--yaw-samples", type=int, default=61)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    records = json.loads(args.inputs.read_text())["records"]
    forecast.PANEL_YAW_SAMPLES = args.yaw_samples
    result = []
    for begin in range(0, len(records), 16):
        chunk = records[begin : begin + 16]
        t = lambda key: torch.as_tensor(
            np.asarray([r[key] for r in chunk]), dtype=torch.float64, device=args.device
        )
        scalar = lambda x: torch.full((len(chunk),), x, dtype=torch.float64, device=args.device)
        root, hinge, tangent = t("root"), t("hinge"), t("tangent")
        bank = forecast.panel_candidate_bank(
            t("desired"),
            t("orientation"),
            t("normal"),
            tangent,
            t("jaw_local"),
            t("depth"),
            hinge,
            root,
            torch.tensor([[0.2, 0.0, 0.7]], device=args.device).expand(len(chunk), -1),
            scalar(0.03),
            t("edge"),
            t("bottom"),
            t("top"),
            scalar(0.05),
            scalar(0.7),
            t("handle_bounds"),
            t("sign"),
        )
        yaw = torch.linspace(-30.0, 30.0, args.yaw_samples, device=args.device, dtype=torch.float64)
        pick = (bank["parameters"][:, :, 2, None] == yaw[None, None]).long().argmax(1)
        gather = lambda key: bank[key].gather(1, pick[..., None].expand(-1, -1, 3))
        target, surface, tip = gather("targets"), gather("tip_surface"), gather("tip_offset")
        radius = ((target - hinge[:, None]) * tangent[:, None]).sum(-1)
        w, x, y, z = root[:, 3:7].unbind(-1)
        heading = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        center = root[:, :3] + _quat_rotate(
            _yaw_quat(heading),
            torch.tensor([[0.2, 0.0, 0.7]], device=args.device, dtype=torch.float64).expand(
                len(chunk), -1
            ),
        )
        delta = torch.deg2rad(torch.arange(46, device=args.device, dtype=torch.float64))
        angle = t("sign")[:, None, None] * delta[None, None]
        c, s = angle.cos(), angle.sin()
        relative = surface - hinge[:, None]
        point = (
            hinge[:, None, None]
            + torch.stack(
                (
                    c * relative[:, :, None, 0] - s * relative[:, :, None, 1],
                    s * relative[:, :, None, 0] + c * relative[:, :, None, 1],
                    relative[:, :, None, 2].expand(-1, -1, len(delta)),
                ),
                -1,
            )
            - tip[:, :, None]
        )
        direction = torch.stack(
            (
                c * tangent[:, None, None, 0] - s * tangent[:, None, None, 1],
                s * tangent[:, None, None, 0] + c * tangent[:, None, None, 1],
            ),
            -1,
        )
        relative = center[:, None, None, :2] - point[..., :2]
        middle = (relative * direction).sum(-1)
        half_sq = 0.7**2 - ((relative * relative).sum(-1) - middle.square())
        half = half_sq.clamp_min(0.0).sqrt()
        bounds = bank["radial_bounds"]
        low = (
            torch.maximum(radius[:, :, None] + middle - half, bounds[:, :, 0, None])
            .cummax(-1)
            .values
        )
        high = (
            torch.minimum(radius[:, :, None] + middle + half, bounds[:, :, 1, None])
            .cummin(-1)
            .values
        )
        feasible = (half_sq >= 0).cumprod(-1).bool() & (low <= high)
        first_bad = torch.where(
            ~feasible, torch.arange(len(delta), device=args.device), len(delta)
        ).amin(-1)
        progress = (first_bad - 1).clamp_min(-1).amax(-1).cpu().tolist()
        for record, value in zip(chunk, progress):
            result.append(dict(seed=record["seed"], continuous_radial_progress_deg=value))
    summary = dict(
        seeds=len(result),
        yaw_samples=args.yaw_samples,
        at_least_5_deg=sum(r["continuous_radial_progress_deg"] >= 5 for r in result),
        no_contact_interval=sum(r["continuous_radial_progress_deg"] < 0 for r in result),
    )
    args.output.write_text(json.dumps(dict(summary=summary, records=result), indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
