"""Current rigid states move shared BVH frames, not room meshes or another row's objects."""

from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.auto_open.motion_math import _euler_xyz_quat
from gr00t.rl.pick.basket_geometry import BasketGeometryBatch
from gr00t.rl.pick.geometry_views import TranslatedGeometry
from gr00t.rl.pick.scene_geometry import SceneGeometry

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU rigid basket scene frames"
)


def test_live_basket_and_distractor_frames_reuse_meshes_and_ignore_absent_padding(monkeypatch):
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], device="cuda")
    plane = torch.tensor([[-2.0, -2, 0], [2.0, -2, 0], [2.0, 2, 0], [-2.0, 2, 0]], device="cuda")
    room = SceneGeometry(plane, faces)
    basket = SimpleNamespace(
        geometry=SceneGeometry(plane * 0.2 + plane.new_tensor([0, 0, 0.2]), faces)
    )
    prop = SceneGeometry(plane * 0.1, faces)
    offsets = plane.new_tensor([[0, 0, 0], [10, 0, 0]])
    controllers = [
        SimpleNamespace(
            static_scene=TranslatedGeometry(room, offsets[row]),
            basket=basket,
            episode={"distractors": [{"asset": {"usd_path": "fruit.usd"}}] if row == 0 else []},
        )
        for row in range(2)
    ]
    loads = []

    def load(path, device):
        loads.append(path)
        return prop

    factory = BasketGeometryBatch(controllers, mesh_loader=load)
    root = plane.new_zeros(2, 13)
    root[:, :3] = offsets + plane.new_tensor([0, 0, 0.52])
    root[:, 3] = 1
    distractors = root[:, None].clone()
    distractors[..., 2] = 0.7
    state = {"root_states": root, "distractor_states": distractors}
    geometry = factory.update(state)
    p = offsets[:, None] + plane.new_tensor([[[0, 0, 0.9]]])
    q = plane.new_tensor([1, 0, 0, 0]).expand(2, 1, -1)
    cloud = plane.new_zeros(1, 3)
    initial = geometry.nonrobot_geometry.rigid_surface_clearance(p, q, cloud)
    torch.testing.assert_close(initial[:, 0], plane.new_tensor([0.2, 0.9]), atol=1e-6, rtol=1e-5)
    mesh_pointer = factory.props[0].mesh_ids.data_ptr()
    source_mesh = factory.props[0].geometries[0]
    root[:, 2] += 0.1
    distractors[0, 0, 3:7] = _euler_xyz_quat(plane.new_tensor([[0, 1.5707963, 0]]))[0]

    def forbidden(*args, **kwargs):
        raise AssertionError("Scene frame updates must not read rigid states on the host")

    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        geometry = factory.update(state)
    moved_p = offsets[:, None] + plane.new_tensor([[[0.1, 0, 0.85]]])
    nonrobot = geometry.nonrobot_geometry.rigid_surface_clearance(moved_p, q, cloud)
    torch.testing.assert_close(nonrobot[:, 0], plane.new_tensor([0.1, 0.85]), atol=2e-6, rtol=1e-5)
    complete = geometry.rigid_surface_clearance(moved_p, q, cloud)
    torch.testing.assert_close(complete[:, 0], plane.new_full((2,), 0.03), atol=2e-6, rtol=1e-5)
    order = torch.tensor([1, 0], device="cuda")
    selected = geometry.select(order)
    assert selected.basket_model is basket
    torch.testing.assert_close(
        selected.rigid_surface_clearance(moved_p[order], q, cloud), complete[order]
    )
    assert factory.props[0].mesh_ids.data_ptr() == mesh_pointer
    assert factory.props[0].geometries[0] is source_mesh is prop
    assert loads == ["fruit.usd"]


def test_target_bvh_reuses_distractor_assets_and_tracks_current_pose_without_host_reads(
    monkeypatch,
):
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], device="cuda")
    points = torch.tensor([[-2.0, -2, 0], [2.0, -2, 0], [2.0, 2, 0], [-2.0, 2, 0]], device="cuda")
    room = SceneGeometry(points - points.new_tensor([0, 0, 2]), faces)
    prop = SceneGeometry(points * 0.1, faces)
    basket = SimpleNamespace(geometry=room)
    offsets = points.new_tensor([[0, 0, 0], [10, 0, 0]])
    controllers = [
        SimpleNamespace(
            static_scene=TranslatedGeometry(room, offsets[row]),
            basket=basket,
            episode={
                "asset": {"usd_path": "fruit.usd"},
                "distractors": [{"asset": {"usd_path": "fruit.usd"}}] if row == 0 else [],
            },
        )
        for row in range(2)
    ]
    loads = []

    def load(path, device):
        loads.append(path)
        return prop

    factory = BasketGeometryBatch(controllers, mesh_loader=load)
    roots = points.new_zeros(2, 13)
    roots[:, :3], roots[:, 3] = offsets, 1
    obj = roots.clone()
    obj[:, 2] = points.new_tensor([0.6, 0.7])
    props = roots[:, None].clone()
    props[..., 0] += 1
    state = {"root_states": roots, "object_state": obj, "distractor_states": props}
    factory.update(state)
    with_target, surface = factory.with_target(state)
    assert loads == ["fruit.usd"]
    assert surface.geometries[0] is factory.props[0].geometries[0] is prop
    p = offsets[:, None] + points.new_tensor([[[0, 0, 0.9]]])
    q = points.new_tensor([1, 0, 0, 0]).expand(2, 1, -1)
    cloud = points.new_zeros(1, 3)
    torch.testing.assert_close(
        surface.rigid_surface_clearance(p, q, cloud)[:, 0], points.new_tensor([0.3, 0.2])
    )
    obj[:, 2] += 0.1

    def forbidden(*args, **kwargs):
        raise AssertionError("Target frames must update on the GPU without rebuilding meshes")

    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        factory.update(state)
        updated, same_surface = factory.with_target(state)
    assert same_surface is surface
    assert loads == ["fruit.usd"]
    expected = points.new_tensor([[0.2], [0.1]])
    torch.testing.assert_close(updated.rigid_surface_clearance(p, q, cloud), expected)
    order = torch.tensor([1, 0], device="cuda")
    torch.testing.assert_close(
        updated.select(order).rigid_surface_clearance(p[order], q, cloud), expected[order]
    )
