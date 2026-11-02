"""Group only static-scene selection; episode randomization keeps its own seed."""

import numpy as np

EPISODES_PER_SCENE = 20


def select_pick_scene(records, seed):
    records = sorted(records, key=lambda record: record.scene_id)
    if not records:
        raise ValueError("Cannot select a pick scene from an empty catalog")
    scene_seed = int(seed) // EPISODES_PER_SCENE
    cycle, index = divmod(scene_seed, len(records))
    random = np.random.default_rng(np.random.SeedSequence([cycle, 0x5049434B]))
    return records[int(random.permutation(len(records))[index])]


def scene_selection_metadata(seed, *, fixed=False):
    return {
        "policy": "fixed" if fixed else "seed_group",
        "scene_seed": int(seed) // EPISODES_PER_SCENE,
        "episodes_per_scene": EPISODES_PER_SCENE,
    }


class PreparationSceneCache:
    """Reuse one immutable GPU scene across a seed group without keeping all rooms."""

    def __init__(self):
        self.path = None
        self.geometry = None

    def get(self, record, cache_root):
        from gr00t.rl.pick.scene_geometry import read_scene_triangles, static_scene_geometry

        path = str(record.usd_path)
        if path != self.path:
            self.geometry = None
            self.path = None
            points, faces, labels = read_scene_triangles(
                record.usd_path, cache_root, return_labels=True
            )
            self.geometry = static_scene_geometry(
                points, faces, labels=labels, ground_plane_z_m=0.0
            )
            self.path = path
        return self.geometry


class AdmissionSceneCache:
    """Borrow already-loaded, scene-local BVHs; admission must never rebuild a room."""

    def __init__(self, episodes, geometries):
        self.geometries = {}
        for episode, geometry in zip(episodes, geometries, strict=True):
            self.geometries[str(episode["scene_usd_path"])] = geometry.source

    def get(self, record, cache_root):
        return self.geometries[str(record.usd_path)]


def disable_embedded_physics_scenes(stage, scene_roots):
    """Inspect each unique USD hierarchy once, then author only the necessary overrides."""
    from pxr import Usd, UsdPhysics

    relative_paths = {}
    for root_path, asset_path in scene_roots:
        root = stage.GetPrimAtPath(root_path)
        if asset_path not in relative_paths:
            relative_paths[asset_path] = [
                prim.GetPath().MakeRelativePath(root.GetPath())
                for prim in Usd.PrimRange(root)
                if prim.IsA(UsdPhysics.Scene)
            ]
        for relative in relative_paths[asset_path]:
            stage.GetPrimAtPath(root.GetPath().AppendPath(relative)).SetActive(False)
