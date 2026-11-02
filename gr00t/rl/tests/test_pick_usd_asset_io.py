"""Verify the deployed USD reader's non-mmap mode without starting simulation."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.mark.parametrize("real_room", [False, True])
def test_usdc_pread_preserves_composed_values_without_file_mappings(tmp_path, real_room):
    from pxr import Gf, Usd, UsdGeom

    files = []
    for index in range(0 if real_room else 64):
        path = tmp_path / f"mesh_{index}.usdc"
        stage = Usd.Stage.CreateNew(str(path))
        mesh = UsdGeom.Mesh.Define(stage, "/Mesh")
        mesh.CreatePointsAttr([Gf.Vec3f(index, 0, 0), Gf.Vec3f(0, 1, 0), Gf.Vec3f(0, 0, 1)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        stage.GetRootLayer().Save()
        files.append(str(path))
    if real_room:
        path = Path("/datasets/vlnverse_scenes/kujiale_0237/start_result_navigation.usd")
        if not path.is_file():
            pytest.skip("Real VLNVerse room is not mounted")
        files = [str(path)]
    code = textwrap.dedent("""
        import hashlib, json, sys
        from pathlib import Path
        from pxr import Usd
        stages = [Usd.Stage.Open(path) for path in sys.argv[1:]]
        values = [stage.Flatten().ExportToString() for stage in stages]
        assets = {layer.realPath for stage in stages for layer in stage.GetUsedLayers()
                  if layer.realPath}
        maps = Path("/proc/self/maps").read_text().splitlines()
        mapped = sum(any(path in line for path in assets) for line in maps)
        print(json.dumps({"hash": hashlib.sha256("".join(values).encode()).hexdigest(),
                          "mapped_assets": mapped, "stages": len(stages)}))
    """)
    reports = []
    for mode in ("0", "1"):
        result = subprocess.run(
            [sys.executable, "-c", code, *files],
            env={**os.environ, "USDC_USE_PREAD": mode},
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        reports.append(json.loads(result.stdout.splitlines()[-1]))
    assert reports[0]["hash"] == reports[1]["hash"]
    print({"real_room": real_room, "mmap": reports[0], "pread": reports[1]})
    assert reports[0]["stages"] == reports[1]["stages"] == len(files)
    assert reports[0]["mapped_assets"] >= len(files)
    assert reports[1]["mapped_assets"] == 0
