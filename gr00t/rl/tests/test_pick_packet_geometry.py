"""Exact packet storage preserves the shared GPU scene query interface."""

import pytest
import torch
import trimesh

from gr00t.rl.auto_open.motion_math import _euler_xyz_quat
from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.pick.batched_geometry import BatchedSceneGeometry
from gr00t.rl.pick.geometry_views import TranslatedGeometry
from gr00t.rl.pick.packet_geometry import PacketGeometry
from gr00t.rl.pick.scene_geometry import SceneGeometry, static_scene_geometry

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU scene queries")


def scene_pair():
    mesh = trimesh.creation.box()
    points = torch.tensor(mesh.vertices, dtype=torch.float32, device="cuda:0")
    faces = torch.tensor(mesh.faces, dtype=torch.int32, device="cuda:0")
    faces = torch.cat((faces, faces[:2], torch.zeros_like(faces[:1])))
    labels = torch.arange(len(faces), device="cuda:0", dtype=torch.int8) % 6
    native = SceneGeometry(points, faces, labels=labels)
    packet = static_scene_geometry(points, faces, labels=labels, backend="packet")
    return native, packet


def test_static_scene_factory_keeps_native_control_and_packet_interface():
    native, packet = scene_pair()
    assert isinstance(packet, PacketGeometry)
    assert isinstance(packet, SceneGeometry)
    query = packet.points.new_tensor([[0, 0, 1], [0, 0, 0], [2, 2, 2]])
    torch.testing.assert_close(packet.distances(query), native.distances(query))
    torch.testing.assert_close(packet.visible(query, -query), native.visible(query, -query))
    with pytest.raises(ValueError, match="backend"):
        static_scene_geometry(native.points, native.faces, backend="unknown")


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("rays", [False, True])
@pytest.mark.parametrize("maximum", [0.0021, 2.0])
def test_mixed_batch_preserves_masks_frames_labels_and_selection(
    monkeypatch, signed, rays, maximum
):
    native, packet = scene_pair()
    offsets = native.points.new_tensor([[0, 0, 0], [10, -5, 0.2], [-4, 7, -0.1]])
    frames = _euler_xyz_quat(offsets.new_tensor([[0, 0, 0], [0.1, -0.2, 0.3], [-0.2, 0.3, -0.1]]))
    indices = torch.tensor([0, 1, 0], device=offsets.device)
    old = BatchedSceneGeometry([native, native], indices, offsets, frames)
    new = BatchedSceneGeometry([packet, native], indices, offsets, frames)
    local = offsets.new_tensor([[0, 0, 0], [0.49, 0, 0], [0.51, 0, 0], [0.2, 0.1, 1], [2, 2, 2]])
    positions = offsets[:, None] + _quat_rotate(frames[:, None], local[None])
    orientations = frames[:, None].expand(-1, len(local), -1)
    targets = offsets[:, None] + _quat_rotate(frames[:, None], -local[None])
    mask = torch.ones(3, len(local), device=offsets.device, dtype=torch.bool)
    mask[:, -1] = False
    cloud = offsets.new_tensor([[0, 0, 0], [0.04, 0, 0], [0, 0.02, -0.03], [float("nan"), 0, 0]])
    cloud_mask = torch.tensor([True, True, True, False], device=offsets.device)
    options = dict(
        check_rays=rays,
        signed_distance=signed,
        maximum_distance_m=maximum,
        query_mask=mask,
        local_point_mask=cloud_mask,
    )
    order = torch.tensor([2, 0, 1], device=offsets.device)

    def forbidden(*args, **kwargs):
        raise AssertionError("Queries must remain on the GPU")

    with monkeypatch.context() as guard:
        for name in ("item", "cpu", "tolist"):
            guard.setattr(torch.Tensor, name, forbidden)
        expected_ray, actual_ray = old.rays(positions, targets), new.rays(positions, targets)
        expected_labels, actual_labels = old.last_hit_types.clone(), new.last_hit_types.clone()
        expected = old.rigid_surface_clearance(positions, orientations, cloud, **options)
        actual = new.rigid_surface_clearance(positions, orientations, cloud, **options)
        selected = new.select(order)
        shuffled = selected.rigid_surface_clearance(
            positions[order],
            orientations[order],
            cloud,
            **{**options, "query_mask": mask[order]},
        )
    for actual_value, expected_value in zip(actual_ray, expected_ray):
        torch.testing.assert_close(actual_value, expected_value, atol=5e-6, rtol=1e-5)
    torch.testing.assert_close(actual_labels, expected_labels)
    torch.testing.assert_close(actual, expected, atol=5e-6, rtol=1e-5)
    torch.testing.assert_close(shuffled, actual[order])
    torch.testing.assert_close(actual[:, -1], torch.full_like(actual[:, -1], maximum))


def test_composite_views_dispatch_packet_storage_and_validate_metadata():
    native, packet = scene_pair()
    offsets = native.points.new_tensor([[0, 0, 0], [10, 10, 0]])
    batch = BatchedSceneGeometry.from_views(
        [
            TranslatedGeometry(packet, offsets[0]),
            TranslatedGeometry(native, offsets[1]),
        ]
    )
    positions = offsets[:, None] + offsets.new_tensor([[[0, 0, 1]]])
    torch.testing.assert_close(batch.distances(positions), offsets.new_full((2, 1), 0.5))
    packet.labels = packet.labels[:1]
    with pytest.raises(ValueError, match="labels"):
        BatchedSceneGeometry(
            [packet], torch.zeros(1, device=offsets.device, dtype=torch.long), offsets[:1]
        )


def test_repeated_packet_views_share_converted_surface_labels():
    native, packet = scene_pair()
    origins = native.points.new_zeros(3, 3)
    batch = BatchedSceneGeometry(
        [packet, native, packet], torch.arange(3, device=origins.device), origins
    )
    assert batch.label_storage[0].data_ptr() == batch.label_storage[2].data_ptr()
