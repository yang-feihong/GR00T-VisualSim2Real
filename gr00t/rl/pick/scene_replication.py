"""Reuse SDK parsing for identical static rooms without changing their USD."""


class StaticRoomReplication:
    def __init__(self, stage, groups):
        import omni.physx
        from pxr import UsdUtils

        self.groups = groups
        self.targets = ()
        self.stage_id = UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
        self.interface = omni.physx.get_physx_replicator_interface()
        self.interface.register_replicator(
            self.stage_id, self._exclude, self._replicate, self._rename
        )

    def _exclude(self, stage_id):
        return [path for _, targets in self.groups for path in targets]

    def _replicate(self, stage_id):
        for source, targets in self.groups:
            self.targets = targets
            self.interface.replicate(stage_id, source, len(targets), False, False)
        self.targets = ()

    def _rename(self, path, index):
        return self.targets[index]

    def close(self):
        if self.stage_id is not None:
            self.interface.unregister_replicator(self.stage_id)
            self.stage_id = None


def register_static_room_replication(stage, scene_roots):
    from pxr import Usd, UsdPhysics

    by_asset = {}
    for path, asset in scene_roots:
        by_asset.setdefault(asset, []).append(path)
    groups = []
    for paths in by_asset.values():
        if len(paths) < 2:
            continue
        root = stage.GetPrimAtPath(paths[0])
        if any(
            prim.HasAPI(UsdPhysics.RigidBodyAPI)
            or prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            or prim.IsA(UsdPhysics.Joint)
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
        ):
            continue
        groups.append((paths[0], tuple(paths[1:])))
    return StaticRoomReplication(stage, groups) if groups else None
