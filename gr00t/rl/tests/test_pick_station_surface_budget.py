"""Real-room clearance gates keep their meaning when distant distances are clipped."""

from pathlib import Path

import pytest
import torch

from gr00t.rl.pick.batched_geometry import BatchedSceneGeometry
from gr00t.rl.pick.scene_geometry import SceneGeometry, read_scene_triangles
from gr00t.rl.pick.station_batch import _station_gripper_clearance


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU scene query benchmark")
def test_real_room_surface_query_budget_preserves_station_gates():
    scene = Path("/datasets/vlnverse_scenes/kujiale_0274/start_result_navigation.usd")
    if not scene.is_file():
        pytest.skip("downloaded VLNVerse regression room unavailable")
    cache = Path(__file__).resolve().parents[3] / "datasets/pick/.cache/geometry"
    points, faces = read_scene_triangles(scene, cache)
    room = SceneGeometry(points, faces)
    geometry = BatchedSceneGeometry.from_views([room])
    vertices = torch.as_tensor(points, device="cuda", dtype=torch.float32)
    triangles = vertices[torch.as_tensor(faces, device="cuda", dtype=torch.long)]
    rows = torch.linspace(0, len(triangles) - 1, 32, device="cuda").long()
    sample = triangles[rows]
    normal = torch.cross(sample[:, 1] - sample[:, 0], sample[:, 2] - sample[:, 0], dim=-1)
    normal /= normal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    offsets = vertices.new_tensor([-0.1, -0.0021, -0.001, 0, 0.001, 0.002, 0.0021, 0.1, 1])
    positions = (sample.mean(1)[:, None] + offsets[None, :, None] * normal[:, None]).flatten(0, 1)
    positions = positions[None]
    orientations = positions.new_zeros(*positions.shape[:-1], 4)
    orientations[..., 0] = 1
    local = torch.cartesian_prod(
        vertices.new_tensor([-0.08, 0.08]),
        vertices.new_tensor([-0.04, 0.04]),
        vertices.new_tensor([-0.04, 0.04]),
    )

    def measure(maximum):
        geometry.rigid_surface_clearance(
            positions, orientations, local, maximum_distance_m=maximum
        )
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = geometry.rigid_surface_clearance(
            positions, orientations, local, maximum_distance_m=maximum
        )
        end.record()
        end.synchronize()
        return result, start.elapsed_time(end)

    full, full_ms = measure(2.0)
    capped, capped_ms = measure(0.0021)
    torch.testing.assert_close(full >= 0.002, capped >= 0.002)
    torch.testing.assert_close(full < 0, capped < 0)
    print({"triangles": len(faces), "poses": positions.shape[1],
           "full_query_cuda_ms": full_ms, "capped_query_cuda_ms": capped_ms})

    class Surface:
        requests = 0

        def gripper_clearance(self, selected, p, q, **options):
            self.requests += p.shape[0] * p.shape[1]
            return selected.rigid_surface_clearance(p, q, local, **options)

    repeated = positions[0].repeat_interleave(8, dim=0)[:, None].expand(-1, 3, -1)
    q = orientations[0].repeat_interleave(8, dim=0)
    owner = torch.zeros(len(repeated), device="cuda", dtype=torch.long)
    active = torch.ones(repeated.shape[:-1], device="cuda", dtype=torch.bool)
    surface = Surface()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    deduplicated = _station_gripper_clearance(geometry, surface, owner, repeated, q, active)
    end.record()
    end.synchronize()
    expected = capped[0].repeat_interleave(8)[:, None].expand(-1, 3)
    torch.testing.assert_close(deduplicated, expected, atol=1e-6, rtol=1e-5)
    assert surface.requests <= positions.shape[1]
    print({"original_pose_queries": repeated.shape[0] * repeated.shape[1],
           "unique_pose_queries": surface.requests,
           "deduplicated_cuda_ms": start.elapsed_time(end)})
