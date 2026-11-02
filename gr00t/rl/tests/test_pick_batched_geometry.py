"""Different scenes never share query results, even for colocated local poses."""

import pytest
import torch

from gr00t.rl.auto_open.motion_math import _euler_xyz_quat
from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.pick.batched_geometry import BatchedCompositeGeometry, BatchedSceneGeometry
from gr00t.rl.pick.geometry_views import CompositeGeometry, TranslatedGeometry
from gr00t.rl.pick.scene_geometry import SceneGeometry

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU spatial queries")


def planes():
    points = torch.tensor(
        [[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]], device="cuda", dtype=torch.float32
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], device="cuda")
    return [
        SceneGeometry(points, faces),
        SceneGeometry(points + points.new_tensor([0, 0, 0.5]), faces),
    ]


def test_capped_unsigned_surface_queries_keep_clearance_gates_and_crossing_rays():
    low, high = planes()
    geometry = BatchedSceneGeometry.from_views([low, high])
    height = torch.tensor(
        [-0.10, -0.0021, -0.001, 0.0, 0.001, 0.0019, 0.002, 0.0021, 0.01, 0.1, 1],
        device="cuda",
    )
    positions = torch.zeros(2, len(height), 3, device="cuda")
    positions[..., 2] = height[None] + height.new_tensor([0, 0.5])[:, None]
    orientation = positions.new_zeros(2, len(height), 4)
    orientation[..., 0] = 1
    points = positions.new_tensor([[0, 0, -0.05], [0, 0, 0], [0.01, 0, 0.05]])
    full = geometry.rigid_surface_clearance(positions, orientation, points)
    capped = geometry.rigid_surface_clearance(
        positions, orientation, points, maximum_distance_m=0.0021
    )
    torch.testing.assert_close(full >= 0.002, capped >= 0.002)
    torch.testing.assert_close(full < 0, capped < 0)
    assert (full < 0).any()  # Segment crossings still reject poses far from the surface.
    torch.testing.assert_close(capped, full.clamp_max(0.0021), atol=1e-6, rtol=1e-5)


def test_single_scene_masks_match_batched_queries_and_skip_padded_nan_poses():
    scene = planes()[0]
    positions = torch.zeros(2, 3, 3, device="cuda")
    positions[..., 2] = positions.new_tensor([[0.001, 0.05, 0.2], [0.3, 0.004, 0.1]])
    rotations = positions.new_zeros(2, 3, 4)
    rotations[..., 0] = 1
    points = positions.new_tensor([[0, 0, 0], [0.01, 0, 0.001]])
    active = torch.tensor([[True, False, True], [False, True, False]], device="cuda")
    positions[~active] = torch.nan
    rotations[~active] = torch.nan
    expected = BatchedSceneGeometry.from_views([scene, scene]).rigid_surface_clearance(
        positions, rotations, points, query_mask=active, maximum_distance_m=0.25
    )
    actual = scene.rigid_surface_clearance(
        positions, rotations, points, query_mask=active, maximum_distance_m=0.25
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[~active], actual.new_full((3,), 0.25))
    assert torch.isfinite(actual).all()
    with pytest.raises(ValueError, match="mask must match"):
        scene.rigid_surface_clearance(positions, rotations, points, query_mask=active[:, :1])


def test_basket_arm_clearance_uses_same_mask_for_room_and_single_scene_self_geometry():
    from types import SimpleNamespace

    from gr00t.rl.pick.basket_controller import BasketArmCollisionSamples
    from gr00t.rl.pick.collision import ArmCollisionSamples

    low, high = planes()
    arm = ArmCollisionSamples.__new__(ArmCollisionSamples)
    arm.parts = [(0, low.points.new_tensor([[0, 0, 0], [0.01, 0, 0.001]]))]
    wrapped = BasketArmCollisionSamples(arm)
    geometry = SimpleNamespace(
        nonrobot_geometry=BatchedSceneGeometry.from_views([high, high]),
        basket_model=SimpleNamespace(geometry=low),
    )
    roots = low.points.new_zeros(2, 13)
    roots[:, 2], roots[:, 3] = 0.55, 1
    joints = roots.new_zeros(2, 6)
    mask = torch.tensor([True, False], device="cuda")
    full = wrapped.clearance(geometry, roots, joints, maximum_distance_m=0.25)
    actual = wrapped.clearance(geometry, roots, joints, query_mask=mask, maximum_distance_m=0.25)
    torch.testing.assert_close(actual[mask], full[mask])
    torch.testing.assert_close(actual[~mask], actual.new_tensor([0.25]))


def test_nested_basket_geometry_accepts_empty_compacted_candidate_queries():
    from gr00t.rl.pick.basket_geometry import BatchedBasketPlanningGeometry

    low, high = planes()
    rooms = BatchedSceneGeometry.from_views([low, high])
    obstacles = BatchedSceneGeometry.from_views([high, low])
    geometry = BatchedBasketPlanningGeometry(
        BatchedCompositeGeometry((rooms, obstacles)), rooms, None
    ).select(torch.empty(0, device="cuda", dtype=torch.long))
    positions = low.points.new_empty(0, 1, 3)
    orientations = low.points.new_empty(0, 1, 4)
    points = low.points.new_zeros(2, 3)
    assert geometry.rigid_surface_clearance(positions, orientations, points).shape == (0, 1)
    assert geometry.distances(positions).shape == (0, 1)
    hit, normal, distance = geometry.rays(positions, positions)
    assert hit.shape == normal.shape == (0, 1, 3)
    assert distance.shape == geometry.last_hit_types.shape == (0, 1)
    with pytest.raises(ValueError, match="leading environment"):
        rooms.rigid_surface_clearance(positions, orientations, points)


def test_ray_surface_labels_preserve_faces_frames_selection_and_shared_mesh_storage(monkeypatch):
    low, high = planes()
    unlabeled = planes()[0]
    low.labels = torch.tensor([0, 1], device="cuda", dtype=torch.int8)
    high.labels = torch.tensor([2, 3], device="cuda", dtype=torch.int8)
    meshes = [high, low, high, unlabeled]
    offsets = torch.tensor(
        [[0, 0, 0], [10, 20, 0], [-5, 3, 0.2], [6, 7, 0]], device="cuda", dtype=torch.float32
    )
    frame = _euler_xyz_quat(
        offsets.new_tensor([[0, 0, 0], [0.2, -0.3, 0.4], [0, 0.7, 0], [0, 0, 1]])
    )
    batch = BatchedSceneGeometry(meshes, torch.arange(4, device="cuda"), offsets, frame)
    frame = batch.orientations
    assert batch.face_labels.numel() == 4  # Repeated BVHs do not repeat per-face metadata.
    local = offsets.new_tensor([[[0.4, -0.2, 1], [-0.2, 0.4, 1], [4.5, 0, 1]]]).expand(4, -1, -1)
    world = offsets[:, None] + _quat_rotate(frame[:, None], local)
    end = offsets[:, None] + _quat_rotate(frame[:, None], local - offsets.new_tensor([0, 0, 2]))

    def forbidden(*args, **kwargs):
        raise AssertionError("Ray labels and row selection must remain on device")

    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        hit, normal, distance = batch.rays(world, end)
        labels = batch.last_hit_types.clone()
        order = torch.tensor([2, 0, 3, 1], device="cuda")
        selected = batch.select(order)
        shuffled = selected.rays(world[order], end[order])
    torch.testing.assert_close(
        labels, torch.tensor([[2, 3, 5], [0, 1, 5], [2, 3, 5], [5, 5, 5]], device="cuda")
    )
    torch.testing.assert_close(selected.last_hit_types, labels[order])
    for actual, expected in zip(shuffled, (hit, normal, distance)):
        torch.testing.assert_close(actual, expected[order])
    assert selected.face_labels.data_ptr() == batch.face_labels.data_ptr()
    for row, mesh in enumerate(meshes):
        h, n, d = mesh.rays(local[row], local[row] - local.new_tensor([0, 0, 2]))
        torch.testing.assert_close(
            hit[row], offsets[row] + _quat_rotate(frame[row], h), atol=3e-6, rtol=1e-5
        )
        torch.testing.assert_close(normal[row], _quat_rotate(frame[row], n), atol=3e-6, rtol=1e-5)
        torch.testing.assert_close(distance[row], d, atol=3e-6, rtol=1e-5)
        torch.testing.assert_close(labels[row], mesh.last_hit_types.long())


def test_composite_rays_visibility_distance_and_labels_match_scalar_ragged_layers():
    low, high = planes()
    low.labels = torch.tensor([0, 1], device="cuda")
    high.labels = torch.tensor([2, 3], device="cuda")
    offsets = torch.tensor([[10, 20, 0], [-3, 4, 0]], device="cuda", dtype=torch.float32)
    views = [
        CompositeGeometry(
            (TranslatedGeometry(low, offsets[0]), TranslatedGeometry(high, offsets[0]))
        ),
        TranslatedGeometry(low, offsets[1]),
    ]
    batch = BatchedSceneGeometry.from_views(views)
    assert isinstance(batch, BatchedCompositeGeometry)
    origins = offsets[:, None] + offsets.new_tensor([[[0.4, -0.2, 1], [-0.2, 0.4, 1], [0, 0, 1]]])
    ends = origins.clone()
    ends[..., 2] = origins.new_tensor([-0.1, -0.1, 0.75])
    result = batch.rays(origins, ends)
    labels = batch.last_hit_types.clone()
    visible, distances = batch.visible(origins, ends), batch.distances(origins)
    for row, view in enumerate(views):
        expected = view.rays(origins[row], ends[row])
        for actual, reference in zip(result, expected):
            torch.testing.assert_close(actual[row], reference)
        torch.testing.assert_close(labels[row], view.last_hit_types)
        torch.testing.assert_close(visible[row], view.visible(origins[row], ends[row]))
        torch.testing.assert_close(distances[row], view.distances(origins[row]))
    order = torch.tensor([1, 0], device="cuda")
    selected = batch.select(order)
    for actual, reference in zip(selected.rays(origins[order], ends[order]), result):
        torch.testing.assert_close(actual, reference[order])
    torch.testing.assert_close(selected.last_hit_types, labels[order])
    # A distance tie must retain the first component's label, as in the scalar implementation.
    other = planes()[0]
    other.labels = torch.tensor([4, 4], device="cuda")
    first = BatchedSceneGeometry([low], torch.zeros(2, dtype=torch.long, device="cuda"), offsets)
    second = BatchedSceneGeometry([other], torch.zeros(2, dtype=torch.long, device="cuda"), offsets)
    tied = BatchedCompositeGeometry((first, second))
    tied.rays(origins, ends)
    torch.testing.assert_close(tied.last_hit_types, first.last_hit_types)


def test_ray_metadata_and_query_shape_are_validated_before_launch():
    low = planes()[0]
    origins = torch.zeros(1, 3, device="cuda")
    low.labels = torch.tensor([0], device="cuda")
    with pytest.raises(ValueError, match="labels"):
        BatchedSceneGeometry([low], torch.zeros(1, device="cuda", dtype=torch.long), origins)
    low.labels = None
    batch = BatchedSceneGeometry([low], torch.zeros(1, device="cuda", dtype=torch.long), origins)
    with pytest.raises(ValueError, match="matching shapes"):
        batch.rays(torch.zeros(1, 2, 3, device="cuda"), torch.zeros(1, 1, 3, device="cuda"))


def test_scene_queries_and_selection_are_environment_local():
    scenes = planes()
    offsets = torch.tensor([[0, 0, 0], [20, 30, 0]], device="cuda", dtype=torch.float32)
    batch = BatchedSceneGeometry(scenes, torch.tensor([0, 1], device="cuda"), offsets)
    starts = offsets[:, None] + offsets.new_tensor([[[0, 0, 1], [0.1, 0, 1]]])
    ends = starts - starts.new_tensor([0, 0, 2])
    hits, _, distances = batch.rays(starts, ends)
    torch.testing.assert_close(hits[..., 2], offsets.new_tensor([[0, 0], [0.5, 0.5]]))
    torch.testing.assert_close(distances, offsets.new_tensor([[1, 1], [0.5, 0.5]]))
    torch.testing.assert_close(batch.distances(starts), distances)
    selected = batch.select(torch.tensor([1, 0], device="cuda"))
    torch.testing.assert_close(selected.rays(starts.flip(0), ends.flip(0))[2], distances.flip(0))


def test_batched_surface_results_match_each_single_scene():
    scenes = planes()
    offsets = torch.tensor([[0, 0, 0], [20, 30, 0]], device="cuda", dtype=torch.float32)
    batch = BatchedSceneGeometry(scenes, torch.tensor([0, 1], device="cuda"), offsets)
    positions = offsets[:, None] + offsets.new_tensor([[[0, 0, 1], [0.1, 0, 0.8]]])
    rotations = offsets.new_tensor([1, 0, 0, 0]).expand(2, 2, 4)
    cloud = offsets.new_tensor([[0, 0, 0], [0.1, 0, -0.05]])
    actual = batch.rigid_surface_clearance(positions, rotations, cloud)
    expected = torch.stack(
        [
            scene.rigid_surface_clearance(positions[i] - offsets[i], rotations[i], cloud)
            for i, scene in enumerate(scenes)
        ]
    )
    torch.testing.assert_close(actual, expected)


def test_masked_surface_query_skips_invalid_poses_without_compaction():
    scenes = planes()
    offsets = torch.zeros(2, 3, device="cuda")
    batch = BatchedSceneGeometry(scenes, torch.tensor([0, 1], device="cuda"), offsets)
    positions = offsets[:, None] + offsets.new_tensor([[[0, 0, 1], [0, 0, float("nan")]]])
    rotations = offsets.new_tensor([1, 0, 0, 0]).expand(2, 2, 4)
    cloud = offsets.new_tensor([[0, 0, 0]])
    active = torch.tensor([[True, False], [True, False]], device="cuda")
    actual = batch.rigid_surface_clearance(positions, rotations, cloud, query_mask=active)
    torch.testing.assert_close(actual[:, 0], offsets.new_tensor([1.0, 0.5]))
    torch.testing.assert_close(actual[:, 1], offsets.new_full((2,), 2.0))
    with pytest.raises(ValueError, match="mask"):
        batch.rigid_surface_clearance(positions, rotations, cloud, query_mask=active[:, :1])


def test_composite_batches_reuse_existing_bvhs_and_preserve_ragged_scene_layers():
    scenes = planes()
    offsets = torch.tensor([[20, 30, 0], [50, 60, 0]], device="cuda", dtype=torch.float32)
    views = [
        CompositeGeometry(
            (TranslatedGeometry(scenes[0], offsets[0]), TranslatedGeometry(scenes[1], offsets[0]))
        ),
        TranslatedGeometry(scenes[0], offsets[1]),
    ]
    batch = BatchedSceneGeometry.from_views(views)
    p = offsets[:, None] + offsets.new_tensor([[[0, 0, 0.8]]])
    q = offsets.new_tensor([1, 0, 0, 0]).expand(2, 1, 4)
    points = offsets.new_zeros(1, 3)
    actual = batch.rigid_surface_clearance(p, q, points)
    torch.testing.assert_close(actual, offsets.new_tensor([[0.3], [0.8]]))
    selected = batch.select(torch.tensor([1, 0], device="cuda"))
    torch.testing.assert_close(
        selected.rigid_surface_clearance(p.flip(0), q, points), actual.flip(0)
    )
    assert batch.components[0].geometries[0] is scenes[0]
    assert batch.components[1].geometries[0] is scenes[1]


def test_rotated_scene_frames_and_ragged_surface_clouds_match_local_queries(monkeypatch):
    scenes = planes()
    offsets = torch.tensor([[10.0, 20, 0.2], [-3.0, 4, 1]], device="cuda")
    frames = _euler_xyz_quat(offsets.new_tensor([[0, 1.5707963, 0], [1.0471976, 0, 0.7]]))
    batch = BatchedSceneGeometry(scenes, torch.arange(2, device="cuda"), offsets, frames)
    frames = batch.orientations
    local = offsets.new_tensor([[[0, 0, 1], [0.1, 0, 0.8]]]).expand(2, -1, -1)
    world = offsets[:, None] + _quat_rotate(frames[:, None], local)
    ends = offsets[:, None] + _quat_rotate(frames[:, None], local - offsets.new_tensor([0, 0, 2]))
    hit, normal, distance = batch.rays(world, ends)
    for row, scene in enumerate(scenes):
        expected_hit, expected_normal, expected_distance = scene.rays(
            local[row], local[row] - local.new_tensor([0, 0, 2])
        )
        torch.testing.assert_close(
            hit[row], offsets[row] + _quat_rotate(frames[row], expected_hit), atol=3e-6, rtol=1e-5
        )
        torch.testing.assert_close(
            normal[row], _quat_rotate(frames[row], expected_normal), atol=3e-6, rtol=1e-5
        )
        torch.testing.assert_close(distance[row], expected_distance, atol=3e-6, rtol=1e-5)
    torch.testing.assert_close(batch.distances(world), distance, atol=3e-6, rtol=1e-5)
    cloud = offsets.new_tensor(
        [
            [[0, 0, 0], [0.1, 0, -0.1], [float("nan")] * 3],
            [[0, 0, -0.2], [-0.1, 0, -0.1], [0.1, 0, 0]],
        ]
    )
    active = torch.tensor([[True, True, False], [True, True, True]], device="cuda")
    quats = frames[:, None].expand(-1, 2, -1)

    def forbidden(*args, **kwargs):
        raise AssertionError("Rigid BVH frames and point masks must remain on the GPU")

    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        actual = batch.rigid_surface_clearance(world, quats, cloud, local_point_mask=active)
    for row, scene in enumerate(scenes):
        expected = scene.rigid_surface_clearance(
            local[row], local.new_tensor([1, 0, 0, 0]).expand(2, -1), cloud[row, active[row]]
        )
        torch.testing.assert_close(actual[row], expected, atol=3e-6, rtol=1e-5)
    order = torch.tensor([1, 0], device="cuda")
    selected = batch.select(order).rigid_surface_clearance(
        world[order], quats[order], cloud[order], local_point_mask=active[order]
    )
    torch.testing.assert_close(selected, actual[order], atol=3e-6, rtol=1e-5)
