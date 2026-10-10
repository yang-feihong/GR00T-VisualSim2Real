from __future__ import annotations

import re
from types import SimpleNamespace

import torch


class _GroupedArticulationData:
    def __init__(self, owner):
        self._owner = owner

    def __getattr__(self, name):
        owner = self._owner
        values = [getattr(asset.data, name) for asset in owner.assets]
        first = values[0]
        if not torch.is_tensor(first):
            return first
        if first.ndim == 0 or first.shape[0] != len(owner.group_env_ids[0]):
            return first
        output = first.new_empty((owner.num_instances, *first.shape[1:]))
        for env_ids, value in zip(owner.group_env_ids, values, strict=True):
            output[env_ids.to(value.device)] = value
        return output


class GroupedArticulation:
    """Present homogeneous articulation groups as one global env-indexed asset."""

    def __init__(self, assets, expected_group_env_ids, num_instances):
        if not assets:
            raise ValueError("GroupedArticulation requires at least one asset")
        self.assets = list(assets)
        self._expected_group_env_ids = [tuple(map(int, ids)) for ids in expected_group_env_ids]
        self.num_instances = int(num_instances)
        self.data = _GroupedArticulationData(self)
        self._group_env_ids = None
        self._root_physx_view = None

    @property
    def group_env_ids(self):
        if self._group_env_ids is None:
            groups = []
            for asset, expected in zip(
                self.assets, self._expected_group_env_ids, strict=True
            ):
                paths = list(asset.root_physx_view.prim_paths)
                parsed = []
                for path in paths:
                    match = re.search(r"/env_(\d+)/", str(path))
                    if match is None:
                        raise RuntimeError(
                            f"Cannot recover environment id from articulation path {path!r}"
                        )
                    parsed.append(int(match.group(1)))
                if sorted(parsed) != sorted(expected):
                    raise RuntimeError(
                        "Grouped door articulation paths do not match their configured "
                        f"environment ids: paths={parsed}, expected={list(expected)}"
                    )
                groups.append(torch.tensor(parsed, dtype=torch.long, device=asset.device))
            flat = torch.cat(groups).detach().cpu().tolist()
            if sorted(flat) != list(range(self.num_instances)):
                raise RuntimeError(f"Grouped door mapping is incomplete: {flat}")
            self._group_env_ids = groups
        return self._group_env_ids

    @property
    def root_physx_view(self):
        if self._root_physx_view is None:
            paths = [None] * self.num_instances
            for asset, env_ids in zip(self.assets, self.group_env_ids, strict=True):
                for env_id, path in zip(
                    env_ids.detach().cpu().tolist(),
                    asset.root_physx_view.prim_paths,
                    strict=True,
                ):
                    paths[env_id] = path
            self._root_physx_view = SimpleNamespace(prim_paths=paths)
        return self._root_physx_view

    @property
    def device(self):
        return self.assets[0].device

    @property
    def is_initialized(self):
        return all(asset.is_initialized for asset in self.assets)

    def _selections(self, env_ids):
        device = self.assets[0].device
        if env_ids is None:
            selected = torch.arange(self.num_instances, device=device)
        else:
            selected = torch.as_tensor(env_ids, dtype=torch.long, device=device).reshape(-1)
        for asset, group_ids in zip(self.assets, self.group_env_ids, strict=True):
            matches = selected[:, None] == group_ids[None, :]
            source_rows, local_ids = torch.nonzero(matches, as_tuple=True)
            if source_rows.numel():
                yield asset, source_rows, local_ids

    def reset(self, env_ids=None):
        for asset, _, local_ids in self._selections(env_ids):
            asset.reset(local_ids)

    def write_data_to_sim(self):
        for asset in self.assets:
            asset.write_data_to_sim()

    def update(self, dt):
        for asset in self.assets:
            asset.update(dt)

    def find_bodies(self, *args, **kwargs):
        return self.assets[0].find_bodies(*args, **kwargs)

    def prim_path_for_env(self, env_id):
        env_id = int(env_id)
        for asset, env_ids in zip(self.assets, self.group_env_ids, strict=True):
            matches = torch.nonzero(env_ids == env_id, as_tuple=False).flatten()
            if matches.numel() == 1:
                root_path = str(asset.root_physx_view.prim_paths[int(matches[0])])
                marker = f"/env_{env_id}/"
                suffix = root_path.split(marker, 1)[1]
                return marker.join((root_path.split(marker, 1)[0], suffix.split("/", 1)[0]))
        raise KeyError(f"No grouped door articulation for environment {env_id}")

    def write_root_pose_to_sim(self, root_pose, env_ids=None):
        for asset, rows, local_ids in self._selections(env_ids):
            asset.write_root_pose_to_sim(root_pose[rows], env_ids=local_ids)

    def write_root_velocity_to_sim(self, root_velocity, env_ids=None):
        for asset, rows, local_ids in self._selections(env_ids):
            asset.write_root_velocity_to_sim(root_velocity[rows], env_ids=local_ids)

    def write_joint_state_to_sim(
        self, joint_position, joint_velocity, joint_ids=None, env_ids=None
    ):
        for asset, rows, local_ids in self._selections(env_ids):
            asset.write_joint_state_to_sim(
                joint_position[rows],
                joint_velocity[rows],
                joint_ids=joint_ids,
                env_ids=local_ids,
            )

    def set_joint_position_target(self, target, joint_ids=None, env_ids=None):
        for asset, rows, local_ids in self._selections(env_ids):
            asset.set_joint_position_target(
                target[rows], joint_ids=joint_ids, env_ids=local_ids
            )
