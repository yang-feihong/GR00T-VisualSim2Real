"""Prepare one rank's episodes before Kit starts, in one preprocessing process."""

import argparse
import copy
import json
import sys
from pathlib import Path

LOWLEVEL_ROOT = Path(__file__).resolve().parents[1] / "third_party/visual_wholebody_lowlevel"
sys.path.insert(0, str(LOWLEVEL_ROOT))

from gr00t.rl.pick.basket_preparation import prepare_basket_batch  # noqa: E402
from gr00t.rl.pick.external_preparation import prepare_external_batch  # noqa: E402
from gr00t.rl.pick.prepare import NoValidInitialScene  # noqa: E402
from gr00t.rl.pick.scene_selection import PreparationSceneCache  # noqa: E402


def local_episode_descriptor(episode):
    """Serialized canonical episodes use world coordinates; spawning adds origins."""
    episode = copy.deepcopy(episode)
    for key in ("physics_scene_usd_path", "planning_scene_usd_path", "scene_scope"):
        episode.pop(key, None)
    origin = episode.pop("batch_scene_origin_m", None)
    if origin is not None:
        for descriptor in (episode, *episode.get("distractors", [])):
            descriptor["object_position"] = [
                value - offset for value, offset in zip(descriptor["object_position"], origin)
            ]
            descriptor["support_z"] -= origin[2]
        for key in ("base_xy", "station_xy"):
            episode[key] = [value - offset for value, offset in zip(episode[key], origin[:2])]
    return episode


def main():
    from gr00t.rl.batch_workflow import bind_rank_parent_lifetime

    bind_rank_parent_lifetime()
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--task", choices=("external", "basket"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes-root", type=Path, default=Path("/datasets/vlnverse_scenes"))
    parser.add_argument("--episodes-manifest", type=Path)
    parser.add_argument("--support", default="any")
    parser.add_argument("--scene-id")
    parser.add_argument("--asset")
    args = parser.parse_args()
    print(
        f"[pick:prepare] source_root={Path(__file__).resolve().parents[3]} "
        f"task={args.task} seeds={len(args.seeds)}",
        flush=True,
    )
    supplied = json.loads(args.episodes_manifest.read_text()) if args.episodes_manifest else {}
    rows = []
    scene_cache = PreparationSceneCache()
    requested = [seed for seed in args.seeds if str(seed) not in supplied]
    prepare = prepare_external_batch if args.task == "external" else prepare_basket_batch
    options = {"scene_id": args.scene_id, "asset_name": args.asset}
    if args.task == "external":
        options["support"] = args.support
    prepared = (
        prepare(
            requested,
            args.scenes_root,
            scene_cache,
            **options,
        )
        if requested
        else {}
    )
    for seed in args.seeds:
        output = args.output.parent / f"seed_{seed}/episode.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            if str(seed) in supplied:
                supplied_row = json.loads(Path(supplied[str(seed)]).read_text())
                if "rejection" in supplied_row:
                    if int(supplied_row["seed"]) != seed:
                        raise ValueError("Explicit rejection seed differs from its rank assignment")
                    rows.append({"seed": seed, "rejection": supplied_row["rejection"]})
                    output.write_text(json.dumps(rows[-1], indent=2) + "\n")
                    continue
                episode = local_episode_descriptor(supplied_row.get("episode", supplied_row))
                if int(episode["seed"]) != seed:
                    raise ValueError("Explicit episode seed differs from its rank assignment")
                output.write_text(json.dumps(episode, indent=2) + "\n")
            else:
                episode = prepared[seed]
                if isinstance(episode, NoValidInitialScene):
                    raise episode
            output.write_text(json.dumps(episode, indent=2) + "\n")
            rows.append({"seed": seed, "episode": episode})
        except NoValidInitialScene as error:
            rows.append(
                {"seed": seed, "rejection": {"reason": str(error), "details": error.details}}
            )
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
