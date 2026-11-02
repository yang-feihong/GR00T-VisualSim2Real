"""Static-source eligibility and callback ownership, without starting Kit."""

from types import SimpleNamespace

from gr00t.rl.pick import scene_replication


def test_only_identical_static_rooms_are_replicated(monkeypatch):
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    roots = []
    for name, asset in (
        ("A", "static"),
        ("B", "static"),
        ("C", "single"),
        ("D", "dynamic"),
        ("E", "dynamic"),
        ("F", "joint"),
        ("G", "joint"),
    ):
        root = UsdGeom.Xform.Define(stage, f"/{name}")
        roots.append((str(root.GetPath()), asset))
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath("/D"))
    UsdPhysics.FixedJoint.Define(stage, "/F/Joint")
    monkeypatch.setattr(scene_replication, "StaticRoomReplication", lambda stage, groups: groups)
    assert scene_replication.register_static_room_replication(stage, roots) == [("/A", ("/B",))]
    assert scene_replication.register_static_room_replication(stage, roots[2:]) is None


def test_replication_callbacks_keep_each_group_targets_and_release_once():
    replication = object.__new__(scene_replication.StaticRoomReplication)
    replication.groups = [("/SourceA", ("/A0", "/A1")), ("/SourceB", ("/B0",))]
    replication.targets = ()
    replication.stage_id = 12
    copied, released = [], []

    def replicate(stage_id, source, count, env_ids, fabric):
        assert (stage_id, env_ids, fabric) == (12, False, False)
        copied.append((source, [replication._rename(source, index) for index in range(count)]))

    replication.interface = SimpleNamespace(
        replicate=replicate, unregister_replicator=released.append
    )
    assert replication._exclude(12) == ["/A0", "/A1", "/B0"]
    replication._replicate(12)
    assert copied == [("/SourceA", ["/A0", "/A1"]), ("/SourceB", ["/B0"])]
    assert replication.targets == ()
    replication.close()
    replication.close()
    assert released == [12]
