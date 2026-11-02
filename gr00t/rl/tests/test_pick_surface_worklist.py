"""Compare conservative block rejection against every original surface point."""

import pytest
import torch
import trimesh

from gr00t.rl.pick import surface_worklist as worklist
from gr00t.rl.pick.batched_geometry import BatchedSceneGeometry
from gr00t.rl.pick.scene_geometry import SceneGeometry

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU BVH")


def full_worklist(
    meshes,
    offsets,
    frames,
    stride,
    positions,
    orientations,
    points,
    local_stride,
    point_active,
    active,
    maximum,
    check_rays,
    **kwargs,
):
    samples = local_stride or len(points)
    blocks = (samples + worklist.SURFACE_BLOCK_SIZE - 1) // worklist.SURFACE_BLOCK_SIZE
    poses = torch.where(active != 0)[0]
    return (poses[:, None] * blocks + torch.arange(blocks, device=poses.device)).flatten(), blocks


@pytest.mark.parametrize("rays", [False, True])
@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("cap", [0.0021, 2.0])
def test_block_rejection_matches_full_original_cloud(monkeypatch, rays, signed, cap):
    mesh = trimesh.creation.box()
    points = torch.tensor(mesh.vertices, device="cuda", dtype=torch.float32)
    scene = SceneGeometry(points, torch.tensor(mesh.faces, device="cuda", dtype=torch.int32))
    batch = BatchedSceneGeometry.from_views([scene, scene])
    generator = torch.Generator(device="cuda").manual_seed(67)
    cloud = torch.randn(769, 3, device="cuda", generator=generator) * 0.08
    cloud[256:512] += 10
    # Endpoint is far beyond the cap, but its TCP ray crosses the original box.
    cloud[-1] = cloud.new_tensor([-3, 0, 0])
    poses = points.new_tensor(
        [[[0.499, 0, 0], [1, 0, 0], [10, 10, 10]], [[0, 0, 0], [0, 0, 0.501], [-8, 5, 2]]]
    )
    rotations = points.new_tensor([1, 0, 0, 0]).expand(2, 3, 4).contiguous()
    options = dict(check_rays=rays, signed_distance=signed, maximum_distance_m=cap)
    with torch.inference_mode():
        actual_batch = batch.rigid_surface_clearance(poses, rotations, cloud, **options)
        actual_scene = scene.rigid_surface_clearance(poses, rotations, cloud, **options)
        with monkeypatch.context() as scope:
            scope.setattr(worklist, "surface_worklist", full_worklist)
            expected_batch = batch.rigid_surface_clearance(poses, rotations, cloud, **options)
            expected_scene = scene.rigid_surface_clearance(poses, rotations, cloud, **options)
    torch.testing.assert_close(actual_batch, expected_batch, rtol=0, atol=0)
    torch.testing.assert_close(actual_scene, expected_scene, rtol=0, atol=0)


def test_cached_bounds_follow_source_mutation():
    points = torch.zeros(513, 3, device="cuda")
    active = torch.ones(513, device="cuda", dtype=torch.int32)
    first = worklist._bounds(points, active, 0, None)
    points += 3
    second = worklist._bounds(points, active, 0, None)
    torch.testing.assert_close(second[0], first[0] + 3)
