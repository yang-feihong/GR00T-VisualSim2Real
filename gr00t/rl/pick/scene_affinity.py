"""Build immutable scene-affinity metadata, without task outcomes or simulation."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np


def triangle_count(episode):
    path = Path(episode["scene_usd_path"])
    identity = f"{path}:{path.stat().st_mtime_ns}:collision-v4-labels"
    cache = Path(episode["geometry_cache_root"]) / (
        hashlib.sha256(identity.encode()).hexdigest()[:20] + ".npz"
    )
    with zipfile.ZipFile(cache) as archive, archive.open("faces.npy") as header:
        version = np.lib.format.read_magic(header)
        shape, _, _ = (
            np.lib.format.read_array_header_1_0(header)
            if version == (1, 0)
            else np.lib.format.read_array_header_2_0(header)
        )
    if len(shape) != 2 or shape[1] != 3:
        raise ValueError(f"Invalid original scene triangle cache: {cache}")
    return shape[0]


def build_manifests(prepared, seeds, output):
    """Snapshot every requested row, including initial scene rejection rows."""
    from gr00t.rl.task_runtime import BatchIdentity

    seeds = BatchIdentity(tuple(seeds)).seeds
    records = {}
    for path in prepared:
        for row in json.loads(Path(path).read_text()):
            seed = int(row["seed"])
            if seed in records and records[seed] != row:
                raise ValueError(f"Conflicting prepared scene descriptors for seed {seed}")
            records[seed] = row
    missing = set(seeds) - records.keys()
    if missing:
        raise ValueError(f"Missing prepared seed descriptors: {sorted(missing)}")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable scene manifest: {output}")
    affinity, paths, weights = {}, {}, {}
    for seed in seeds:
        row = records[seed]
        if "episode" in row:
            episode = row["episode"]
            if int(episode["seed"]) != seed:
                raise ValueError(f"Prepared episode differs from its seed key: {seed}")
            key = str(Path(episode["scene_usd_path"]).resolve())
            if key not in weights:
                weights[key] = triangle_count(episode)
            affinity[str(seed)] = {"key": key, "weight": weights[key]}
        elif "rejection" in row:
            affinity[str(seed)] = {"key": "initial_scene_rejection", "weight": 0}
        else:
            raise ValueError(f"Prepared seed has neither episode nor rejection: {seed}")
    output.mkdir(parents=True)
    episodes = output / "episodes"
    episodes.mkdir()
    for seed in seeds:
        destination = episodes / f"seed_{seed}.json"
        destination.write_text(json.dumps(records[seed], indent=2) + "\n")
        paths[str(seed)] = str(destination.resolve())
    (output / "affinity.json").write_text(json.dumps(affinity, indent=2) + "\n")
    (output / "episodes.json").write_text(json.dumps(paths, indent=2) + "\n")
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "seed_count": len(seeds),
                "scene_count": len(weights),
                "prepared_sources": [str(Path(path).resolve()) for path in prepared],
                "weight_definition": "original static scene triangle count; no task outcomes",
                "initial_rejections": sum("rejection" in records[seed] for seed in seeds),
            },
            indent=2,
        )
        + "\n"
    )
    return affinity


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--prepared", type=Path, nargs="+", required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-end", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    affinity = build_manifests(
        args.prepared, range(args.seed_start, args.seed_end + 1), args.output
    )
    print(json.dumps({"output": str(args.output), "seeds": len(affinity)}), flush=True)


if __name__ == "__main__":
    main()
