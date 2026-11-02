"""Prepare seed-scoped, spatially validated pick episodes without starting Isaac Sim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


class NoValidInitialScene(RuntimeError):
    """A geometric sample batch had no admissible placement."""

    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}


def reject_preparation(output, error):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_name("preparation_rejection.json").write_text(
        json.dumps({"reason": str(error), "details": error.details}, indent=2) + "\n"
    )
    raise SystemExit(3)


def prepare_episode(
    seed,
    scenes_root,
    output,
    *,
    scene_id=None,
    support="any",
    asset_name=None,
    distractor_assets=None,
    fixed_asset=None,
    scene_cache=None,
):
    """Single-seed tooling delegates to the same rank-batched preparation."""
    from gr00t.rl.pick.external_preparation import prepare_external_batch
    from gr00t.rl.pick.scene_selection import PreparationSceneCache

    results = prepare_external_batch(
        [seed],
        scenes_root,
        scene_cache or PreparationSceneCache(),
        scene_id=scene_id,
        support=support,
        asset_name=asset_name,
        fixed_asset=fixed_asset,
        distractor_assets=distractor_assets,
    )
    episode = results[seed]
    if isinstance(episode, NoValidInitialScene):
        raise episode
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(episode, indent=2) + "\n")
    return episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scenes-root", type=Path, default=Path("/datasets/vlnverse_scenes"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-id")
    parser.add_argument(
        "--support",
        choices=("any", "floor", "furniture", "table", "chair", "sofa", "bed"),
        default="any",
    )
    parser.add_argument(
        "--asset", help="Installed scan instance name; defaults to category-balanced sampling"
    )
    args = parser.parse_args()
    try:
        prepare_episode(
            args.seed,
            args.scenes_root,
            args.output,
            scene_id=args.scene_id,
            support=args.support,
            asset_name=args.asset,
        )
    except NoValidInitialScene as error:
        reject_preparation(args.output, error)


if __name__ == "__main__":
    main()
