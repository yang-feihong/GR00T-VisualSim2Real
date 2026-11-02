"""Audit recorded PhysX queries independently of Kit's process shutdown."""

import json
import os
from pathlib import Path

import pytest


@pytest.mark.skipif(
    not os.environ.get("PICK_REPLICATION_BASELINE"), reason="Recorded queries required"
)
def test_all_recorded_collision_hits_and_positions_are_exact(tmp_path):
    from pxr import Sdf, Usd, UsdShade

    baseline = json.loads(Path(os.environ["PICK_REPLICATION_BASELINE"]).read_text())
    candidate = json.loads(Path(os.environ["PICK_REPLICATION_CANDIDATE"]).read_text())
    assert not baseline["replication_enabled"] and candidate["replication_enabled"]
    assert baseline["assets"] == candidate["assets"]
    assert baseline["query_count"] == candidate["query_count"] == 5200
    aliases = []
    for group, (old_group, new_group) in enumerate(
        zip(baseline["queries"], candidate["queries"], strict=True)
    ):
        stage = Usd.Stage.Open(baseline["assets"][group])
        root = stage.GetDefaultPrim().GetPath()
        for copy, (old_row, new_row) in enumerate(zip(old_group, new_group, strict=True)):
            for request, (old, new) in enumerate(zip(old_row, new_row, strict=True)):
                assert old["hit"] == new["hit"]
                assert old["position"] == new["position"]
                if old["collision"] == new["collision"]:
                    continue
                assert old["collision"].rsplit("/", 1)[0] == new["collision"].rsplit("/", 1)[0]
                materials = []
                for name in (old["collision"], new["collision"]):
                    prim = stage.GetPrimAtPath(
                        root.AppendPath(Sdf.Path(name).MakeRelativePath(Sdf.Path.absoluteRootPath))
                    )
                    assert prim.IsValid()
                    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
                    materials.append(str(material.GetPath()))
                assert materials[0] == materials[1]
                aliases.append(
                    {
                        "group": group,
                        "copy": copy,
                        "request": request,
                        "old": old["collision"],
                        "new": new["collision"],
                        "physics_material": materials[0],
                    }
                )
    report = {
        "validated": True,
        "query_count": 5200,
        "exact_hit_flags_and_positions": True,
        "coincident_submesh_aliases": aliases,
    }
    (tmp_path / "recorded_replication_validation.json").write_text(json.dumps(report, indent=2))
