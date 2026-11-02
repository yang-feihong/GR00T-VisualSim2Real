"""Shared immutable room/object BVHs with live scene-owned rigid frames."""

from pathlib import Path

import numpy as np
import torch

from gr00t.rl.pick.batched_geometry import BatchedCompositeGeometry, BatchedSceneGeometry
from gr00t.rl.pick.scene_geometry import SceneGeometry
from gr00t.rl.third_party.visual_wholebody_lowlevel.legged_gym.utils.batch_invariant_math import (
    fixed_normalize,
)


def load_collider_mesh(path, device):
    directory = Path(path).parent
    points = torch.as_tensor(np.load(directory / "collision_support.npy"), device=device)
    faces = torch.as_tensor(np.load(directory / "collision_faces.npy"), device=device)
    return SceneGeometry(points, faces, device=device)


def bind_rigid_frames(view, positions, orientations):
    view.origins = positions.float().clone().contiguous()
    view.orientations = fixed_normalize(orientations, eps=1e-8).float().contiguous()


class SceneObjectGeometryBatch:
    """Load each object mesh once and update only GPU transforms between plans."""

    _bind = staticmethod(bind_rigid_frames)

    def __init__(self, episodes, rooms, *, mesh_loader=load_collider_mesh):
        self.count, self.episodes, self.rooms = len(episodes), episodes, rooms
        self.mesh_loader, self.target = mesh_loader, None
        device = rooms.device
        origins = torch.zeros(self.count, 3, device=device)
        descriptors = [episode.get("distractors", []) for episode in episodes]
        paths = list(
            dict.fromkeys(item["asset"]["usd_path"] for row in descriptors for item in row)
        )
        meshes = [mesh_loader(path, device) for path in paths]
        self.mesh_cache = dict(zip(paths, meshes))
        indices = {path: index for index, path in enumerate(paths)}
        self.props, self.present = [], []
        for slot in range(max(map(len, descriptors), default=0)):
            ids = torch.tensor(
                [
                    indices[row[slot]["asset"]["usd_path"]] if slot < len(row) else 0
                    for row in descriptors
                ],
                device=device,
            )
            self.props.append(BatchedSceneGeometry(meshes, ids, origins))
            self.present.append(
                torch.tensor([slot < len(row) for row in descriptors], device=device)
            )
        self.nonrobot_geometry = (
            BatchedCompositeGeometry((rooms, *self.props)) if self.props else rooms
        )
        self.geometry = self.nonrobot_geometry

    def update(self, state):
        if state["root_states"].shape[0] != self.count:
            raise ValueError("Scene geometry state must preserve the complete scene batch")
        if self.props and state["distractor_states"].shape[1] < len(self.props):
            raise ValueError("Scene distractor states omit a configured object slot")
        for slot, view in enumerate(self.props):
            pose = state["distractor_states"][:, slot]
            active = self.present[slot][:, None]
            positions = torch.where(active, pose[:, :3], pose.new_tensor([0, 0, -10000]))
            orientations = torch.where(active, pose[:, 3:7], pose.new_tensor([1, 0, 0, 0]))
            self._bind(view, positions, orientations)
        if self.target is not None:
            self._bind(self.target, state["object_state"][:, :3], state["object_state"][:, 3:7])
        return self.nonrobot_geometry

    def with_target(self, state):
        if self.target is None:
            paths = [episode["asset"]["usd_path"] for episode in self.episodes]
            unique = list(dict.fromkeys(paths))
            for path in unique:
                if path not in self.mesh_cache:
                    self.mesh_cache[path] = self.mesh_loader(path, self.rooms.device)
            indices = {path: index for index, path in enumerate(unique)}
            self.target = BatchedSceneGeometry(
                [self.mesh_cache[path] for path in unique],
                torch.tensor([indices[path] for path in paths], device=self.rooms.device),
                torch.zeros(self.count, 3, device=self.rooms.device),
            )
        pose = state["object_state"]
        self._bind(self.target, pose[:, :3], pose[:, 3:7])
        return BatchedCompositeGeometry((self.nonrobot_geometry, self.target)), self.target
