"""Generate simulator-free border examples and exact sampled parameter records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from gr00t.rl.data_collection.fisheye_border import FisheyeBorderAugmenter, STYLES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = Path(__file__).resolve().parents[1] / (
        "third_party/visual_wholebody_lowlevel/legged_gym/scripts/rgb_cameras_dataset.json"
    )
    spec = json.loads(config_path.read_text())["defaults"]
    height, width = spec["height"], spec["width"]
    yy, xx = np.mgrid[:height, :width]
    # A synthetic inspection pattern makes changed scene pixels conspicuous.
    checker = ((xx // 32 + yy // 32) % 2).astype(np.uint8)
    scene = np.stack((70 + 100 * checker, 145 - 40 * checker, 130 + 60 * checker), -1)
    mask = spec["valid_circle_mask"]
    distance = np.hypot(xx - mask["center"][0], yy - mask["center"][1])
    alpha = np.clip((mask["radius"] - distance) / mask["feather"], 0, 1)
    source = np.rint(scene * alpha[..., None]).astype(np.uint8)
    Image.fromarray(source).save(args.output / "source_masked.png")
    selected = {}
    for seed in range(1000):
        augmenter = FisheyeBorderAugmenter(seed, "base_0_rgb", spec)
        selected.setdefault(augmenter.metadata["style"], (seed, augmenter))
        if len(selected) == len(STYLES):
            break
    panels = [("Synthetic source / original mask", Image.fromarray(source))]
    records = []
    for style in STYLES:
        seed, augmenter = selected[style]
        result = augmenter.apply(source)
        valid = distance < mask["radius"]
        assert np.array_equal(result[valid], source[valid])
        Image.fromarray(result).save(args.output / f"{style}_seed{seed}.png")
        panels.append((f"{style} / seed {seed}", Image.fromarray(result)))
        records.append(augmenter.metadata)
    tile_w, tile_h = 384, 318
    sheet = Image.new("RGB", (tile_w * 3, tile_h * 2), (235, 238, 240))
    draw = ImageDraw.Draw(sheet)
    for index, (label, panel) in enumerate(panels):
        x, y = (index % 3) * tile_w, (index // 3) * tile_h
        draw.text((x + 8, y + 8), label, fill=(20, 24, 28))
        sheet.paste(panel.resize((384, 288)), (x, y + 30))
    sheet.save(args.output / "contact_sheet.png")
    if args.reference:
        # Separate reference: no claim that its uncalibrated housing is a mask.
        with Image.open(args.reference) as reference:
            reference.convert("RGB").save(args.output / "real_reference.png")
    (args.output / "parameters.json").write_text(json.dumps(records, indent=2) + "\n")
    print(args.output / "contact_sheet.png")


if __name__ == "__main__":
    main()
