"""External admission and stage planners reuse the same live room/prop BVHs."""

from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.pick.batched_geometry import BatchedSceneGeometry
from gr00t.rl.pick.geometry_views import CompositeGeometry, TranslatedGeometry
from gr00t.rl.pick.scene_geometry import SceneGeometry
from gr00t.rl.pick.scene_object_geometry import SceneObjectGeometryBatch
from gr00t.rl.pick.tensor_planning import TensorPickPlanningCoordinator

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU shared scene frames")


def test_external_geometry_is_reused_by_stage_planning_without_host_state_reads(monkeypatch):
    points = torch.tensor(
        [[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]], device="cuda", dtype=torch.float32
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], device="cuda")
    room = SceneGeometry(points, faces, labels=torch.zeros(2, device="cuda", dtype=torch.long))
    prop = SceneGeometry(points * 0.1, faces)
    offsets = points.new_tensor([[0, 0, 0], [10, 0, 0]])
    rooms = [TranslatedGeometry(room, offset) for offset in offsets]
    episodes = [{"distractors": [{"asset": {"usd_path": "fruit.usd"}}]}, {"distractors": []}]
    loads = []

    def loader(path, device):
        loads.append(path)
        return prop

    factory = SceneObjectGeometryBatch(
        episodes, BatchedSceneGeometry.from_views(rooms), mesh_loader=loader
    )
    roots = points.new_zeros(2, 13)
    roots[:, :3], roots[:, 3] = offsets, 1
    payload = roots[:, None].clone()
    payload[..., 2] = 0.7
    payload[1] = torch.nan
    state = dict(root_states=roots, distractor_states=payload)
    geometry = factory.update(state)
    starts = offsets[:, None] + points.new_tensor([[[0.04, -0.02, 1]]])
    ends = starts.clone()
    ends[..., 2] = -0.1
    hit, _, _ = geometry.rays(starts, ends)
    torch.testing.assert_close(hit[..., 2], points.new_tensor([[0.7], [0]]))
    torch.testing.assert_close(geometry.last_hit_types, torch.tensor([[5], [0]], device="cuda"))
    reference = CompositeGeometry((rooms[0], TranslatedGeometry(prop, payload[0, 0, :3])))
    torch.testing.assert_close(
        geometry.rays(starts, ends)[2][0], reference.rays(starts[0], ends[0])[2]
    )
    controls = []
    for row in range(2):
        view = geometry.select(torch.tensor([row], device="cuda"))
        view.external_geometry_factory = factory
        controls.append(SimpleNamespace(geometry=view))
    planner = TensorPickPlanningCoordinator(controls)
    assert planner.external_geometry is factory and planner.carry_geometry is geometry
    mesh_pointer = factory.props[0].mesh_ids.data_ptr()
    payload[0, 0, 2] = 0.8

    def forbidden(*args, **kwargs):
        raise AssertionError("Live geometry must neither rebuild BVHs nor read poses on the host")

    monkeypatch.setattr(BatchedSceneGeometry, "from_views", forbidden)
    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        current = planner._external_scene_geometry(state)
        moved = current.rays(starts, ends)
    torch.testing.assert_close(moved[0][..., 2], points.new_tensor([[0.8], [0]]))
    assert current is geometry and planner._external_scene_geometry() is geometry
    assert loads == ["fruit.usd"] and mesh_pointer == factory.props[0].mesh_ids.data_ptr()
    order = torch.tensor([1, 0], device="cuda")
    torch.testing.assert_close(
        current.select(order).rays(starts[order], ends[order])[2], moved[2][order]
    )
