"""Planning includes the already-existing physics ground without removing room colliders."""

import pytest
import torch

from gr00t.rl.pick.scene_geometry import SceneGeometry, static_scene_geometry


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU collision queries")
@pytest.mark.parametrize("backend", ["native", "packet"])
def test_static_room_adds_existing_global_ground_and_preserves_obstacles(backend):
    points = torch.tensor(
        [
            [-2, -2, -0.05],
            [2, -2, -0.05],
            [2, 2, -0.05],
            [-2, 2, -0.05],
            [0.6, -1, 0],
            [0.6, 1, 0],
            [0.6, 1, 2],
            [0.6, -1, 2],
            [1, -0.5, 0.1],
            [1.5, -0.5, 0.1],
            [1.5, 0.5, 0.1],
            [1, 0.5, 0.1],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    faces = torch.tensor(
        [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [8, 9, 10], [8, 10, 11]],
        dtype=torch.int32,
        device="cuda",
    )
    labels = torch.tensor([0, 0, 5, 5, 1, 1], dtype=torch.int8, device="cuda")
    original = SceneGeometry(points, faces, labels=labels)
    room = static_scene_geometry(
        points, faces, labels=labels, backend=backend, ground_plane_z_m=0.0
    )
    torch.testing.assert_close(room.points[: len(points)], points, atol=0, rtol=0)
    assert len(room.points) == len(points) + 4
    assert room.faces.numel() == faces.numel() + 6
    if backend == "native":
        torch.testing.assert_close(
            room.faces[: faces.numel()].reshape_as(faces), faces, atol=0, rtol=0
        )
        torch.testing.assert_close(room.labels[:-2], labels, atol=0, rtol=0)
    start = points.new_tensor([[-0.5, 0, 0.2], [1.25, 0, 0.2]])
    end = start.clone()
    end[:, 2] = -0.2
    old_hit = original.rays(start, end)[0]
    hit, normal, _ = room.rays(start, end)
    torch.testing.assert_close(old_hit[:, 2], points.new_tensor([-0.05, 0.1]), atol=1e-6, rtol=0)
    torch.testing.assert_close(hit[:, 2], points.new_tensor([0, 0.1]), atol=1e-6, rtol=0)
    assert (normal[:, 2] > 0.99).all()
    assert room.last_hit_types.tolist() == [0, 1]
    assert not room.visible(points.new_tensor([[0, 0, 1]]), points.new_tensor([[1, 0, 1]])).any()
    local = torch.cartesian_prod(*[points.new_tensor([-0.02, 0.02])] * 3)
    pose = points.new_tensor([[[0, 0, 0.01]]])
    rotation = points.new_tensor([[[1, 0, 0, 0]]])
    assert (original.rigid_surface_clearance(pose, rotation, local) >= 0.002).all()
    assert (room.rigid_surface_clearance(pose, rotation, local) < 0.002).all()
    raw = static_scene_geometry(
        points, faces, labels=labels, backend=backend, ground_plane_z_m=None
    )
    assert raw.faces.numel() == faces.numel()
