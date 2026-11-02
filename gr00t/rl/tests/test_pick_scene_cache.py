"""Geometry caches remain complete when GPU ranks prepare the same room."""

import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from gr00t.rl.pick import scene_geometry


def arrays(marker):
    return (
        np.full((3, 3), marker, dtype=np.float32),
        np.array([[0, 1, 2]], dtype=np.int32),
        np.array([marker], dtype=np.int32),
    )


def test_readers_keep_old_complete_cache_until_new_zip_is_closed(tmp_path, monkeypatch):
    cache = tmp_path / "mesh.npz"
    scene_geometry._write_scene_cache(cache, *arrays(1))
    writing, release = threading.Event(), threading.Event()
    original = np.savez

    def partial(output, **values):
        output.write(b"incomplete archive")
        output.flush()
        writing.set()
        assert release.wait(10)
        output.seek(0)
        output.truncate()
        original(output, **values)

    monkeypatch.setattr(np, "savez", partial)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(scene_geometry._write_scene_cache, cache, *arrays(2))
        try:
            assert writing.wait(10)
            current = scene_geometry._read_scene_cache(cache)
            assert current is not None and np.all(current[0] == 1)
        finally:
            release.set()
        future.result(timeout=10)
    assert np.all(scene_geometry._read_scene_cache(cache)[0] == 2)
    assert not list(tmp_path.glob(".*.tmp"))


def test_failed_cache_write_does_not_truncate_valid_archive(tmp_path, monkeypatch):
    cache = tmp_path / "mesh.npz"
    scene_geometry._write_scene_cache(cache, *arrays(1))

    def fail(output, **values):
        output.write(b"unfinished")
        raise OSError("injected write failure")

    monkeypatch.setattr(np, "savez", fail)
    with pytest.raises(OSError, match="injected"):
        scene_geometry._write_scene_cache(cache, *arrays(2))
    assert np.all(scene_geometry._read_scene_cache(cache)[0] == 1)
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("damage", ["truncated", "missing_labels", "wrong_shape", "npy"])
def test_bad_cache_rebuilds_from_same_source_without_changing_geometry(tmp_path, damage):
    source = tmp_path / "room.usda"
    stage = Usd.Stage.CreateNew(str(source))
    mesh = UsdGeom.Mesh.Define(stage, "/World/floor/mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    stage.GetRootLayer().Save()
    cache_root = tmp_path / "cache"
    expected = scene_geometry.read_scene_triangles(source, cache_root, return_labels=True)
    cache = next(cache_root.glob("*.npz"))
    if damage == "truncated":
        cache.write_bytes(b"unfinished zip archive")
    elif damage == "missing_labels":
        np.savez_compressed(cache, points=expected[0], faces=expected[1])
    elif damage == "wrong_shape":
        np.savez_compressed(cache, points=expected[0], faces=expected[1], labels=np.zeros(2))
    else:
        with cache.open("wb") as output:
            np.save(output, expected[0])
    recovered = scene_geometry.read_scene_triangles(source, cache_root, return_labels=True)
    for left, right in zip(recovered, expected):
        np.testing.assert_array_equal(left, right)
    with np.load(cache) as data:
        assert set(data.files) == {"points", "faces", "labels"}


@pytest.mark.parametrize("writer", [np.savez_compressed, np.savez])
def test_compressed_and_stored_cache_arrays_are_exactly_compatible(tmp_path, writer):
    cache = tmp_path / "mesh.npz"
    points, faces, labels = arrays(2)
    writer(cache, points=points, faces=faces, labels=labels)
    recovered = scene_geometry._read_scene_cache(cache)
    for actual, expected in zip(recovered, (points, faces, labels), strict=True):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)


def test_cache_writer_avoids_cpu_deflate_without_dropping_arrays(tmp_path):
    cache = tmp_path / "mesh.npz"
    scene_geometry._write_scene_cache(cache, *arrays(2))
    with zipfile.ZipFile(cache) as archive:
        assert set(archive.namelist()) == {"points.npy", "faces.npy", "labels.npy"}
        assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())


def test_parallel_cache_misses_flatten_once_and_reuse_published_arrays(tmp_path, monkeypatch):
    source = tmp_path / "room.usda"
    source.touch()
    entered = []

    def flatten(path):
        entered.append(path)
        time.sleep(0.02)
        return arrays(1)

    monkeypatch.setattr(scene_geometry, "_flatten_scene_triangles", flatten)
    barrier = threading.Barrier(4)

    def read():
        barrier.wait(timeout=10)
        return scene_geometry.read_scene_triangles(source, tmp_path / "cache", return_labels=True)

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: read(), range(4)))
    assert entered == [source]
    for result in results:
        np.testing.assert_array_equal(result[0], arrays(1)[0])


def test_four_writer_processes_never_publish_a_partial_zip(tmp_path):
    cache = tmp_path / "mesh.npz"
    scene_geometry._write_scene_cache(cache, *arrays(1))
    script = """
import sys
from pathlib import Path
import numpy as np
from gr00t.rl.pick.scene_geometry import _write_scene_cache
for index in range(32):
    marker = int(sys.argv[2]) * 100 + index
    _write_scene_cache(Path(sys.argv[1]),
                      np.full((128, 3), marker, dtype=np.float32),
                      np.array([[0, 1, 2]], dtype=np.int32),
                      np.array([marker], dtype=np.int32))
"""
    workers = [
        subprocess.Popen([sys.executable, "-c", script, str(cache), str(row)])
        for row in range(4)
    ]
    reads, deadline = 0, time.monotonic() + 60
    try:
        while any(worker.poll() is None for worker in workers):
            assert time.monotonic() < deadline, "Cache writer process did not finish"
            result = scene_geometry._read_scene_cache(cache)
            assert result is not None
            assert np.all(result[0] == result[2][0])
            reads += 1
            time.sleep(0.002)
        assert all(worker.wait() == 0 for worker in workers)
        assert reads > 0
        assert not list(tmp_path.glob(".*.tmp"))
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
            worker.wait()
