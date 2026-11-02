"""Per-body robot BVHs match merged collider triangles without rebuilding per scene."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from pxr import Usd, UsdGeom

from gr00t.rl.auto_open.motion_math import _euler_xyz_quat, _quat_rotate
from gr00t.rl.pick.initialization import RobotOcclusionGeometryBatch
from gr00t.rl.pick.scene_geometry import SceneGeometry
from gr00t.rl.utils.b2z1_asset import ROBOT_USD

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU robot visibility")


def merged_geometry(meshes, states):
    """Independent original algorithm: materialize one world mesh from actual body poses."""
    points, faces, offset = [], [], 0
    for index, mesh in meshes:
        pose = states[index]
        points.append(
            _quat_rotate(pose[None, 3:7].expand(len(mesh.points), -1), mesh.points) + pose[:3]
        )
        faces.append(mesh.faces.reshape(-1, 3) + offset)
        offset += len(mesh.points)
    return SceneGeometry(torch.cat(points), torch.cat(faces), device=states.device)


def plane(points=None):
    if points is None:
        points = [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]
    return SceneGeometry(
        torch.tensor(points, device="cuda", dtype=torch.float32),
        torch.tensor([[0, 1, 2], [0, 2, 3]], device="cuda"),
    )


def test_robot_bvh_updates_and_permutation_match_actual_merged_body_meshes(monkeypatch):
    meshes = [(0, plane()), (1, plane())]
    env = SimpleNamespace(num_envs=3, device=torch.device("cuda:0"))
    factory = RobotOcclusionGeometryBatch(env, meshes=meshes)
    states = torch.zeros(3, 2, 13, device="cuda")
    states[..., 3] = 1
    states[..., 0] = torch.tensor([0, 5, 10], device="cuda")[:, None]
    states[:, 0, 2] = states.new_tensor([0.5, 0.0, -2])
    states[:, 1, 2] = states.new_tensor([0.6, 0.9, -2])
    starts = states.new_tensor([[[0, 0, 1], [0.1, 0.1, 1]]]).expand(3, -1, -1).clone()
    starts[..., 0] += states[:, :1, 0]
    ends = starts.clone()
    ends[..., 2] = states.new_tensor([0.75, -0.1])
    pointers = [view.mesh_ids.data_ptr() for _, view in factory.parts]
    geometry = factory.update(states)
    hits = geometry.rays(starts, ends)
    visible = geometry.visible(starts, ends)
    assert visible[:, 0].tolist() == [True, False, True]
    for row in range(3):
        reference = merged_geometry(meshes, states[row])
        for actual, expected in zip(hits, reference.rays(starts[row], ends[row])):
            torch.testing.assert_close(actual[row], expected)
        torch.testing.assert_close(visible[row], reference.visible(starts[row], ends[row]))
    states[:, 1, 3:7] = _euler_xyz_quat(states.new_tensor([[0, 1.5707963, 0]]).expand(3, -1))
    states[:, 1, 0] += 0.3

    def forbidden(*args, **kwargs):
        raise AssertionError("Robot body visibility updates must stay on device")

    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        moved = factory.update(states).visible(starts, ends)
    for row in range(3):
        torch.testing.assert_close(
            moved[row], merged_geometry(meshes, states[row]).visible(starts[row], ends[row])
        )
    order = torch.tensor([2, 0, 1], device="cuda")
    shuffled = RobotOcclusionGeometryBatch(env, meshes=meshes).update(states[order])
    torch.testing.assert_close(shuffled.visible(starts[order], ends[order]), moved[order])
    assert pointers == [view.mesh_ids.data_ptr() for _, view in factory.parts]
    assert all(view.geometries[0] is meshes[i][1] for i, (_, view) in enumerate(factory.parts))


def test_open_gripper_gap_is_not_filled_by_a_single_body_envelope():
    meshes = [
        (0, plane([[-0.05, -0.05, 0], [-0.01, -0.05, 0], [-0.01, 0.05, 0], [-0.05, 0.05, 0]])),
        (1, plane([[0.01, -0.05, 0], [0.05, -0.05, 0], [0.05, 0.05, 0], [0.01, 0.05, 0]])),
    ]
    states = torch.zeros(1, 2, 13, device="cuda")
    states[..., 2], states[..., 3] = 0.5, 1
    geometry = RobotOcclusionGeometryBatch(
        SimpleNamespace(num_envs=1, device="cuda:0"), meshes=meshes
    ).update(states)
    starts = states.new_tensor([[[0, 0, 1], [0.03, 0, 1]]])
    ends = starts.clone()
    ends[..., 2] = 0
    assert geometry.visible(starts, ends).tolist() == [[True, False]]


def test_deployed_robot_collider_loading_requires_no_dynamic_environment_state(monkeypatch):
    import omni

    source = Usd.Stage.Open(str(ROBOT_USD))
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source))
    stage.DefinePrim("/World/envs/env_0/Robot").GetReferences().AddReference(str(ROBOT_USD))
    module = ModuleType("omni.usd")
    def wrong_context():
        raise AssertionError("Visibility must use the simulation-owned stage")
    module.get_context = wrong_context
    monkeypatch.setitem(sys.modules, "omni.usd", module)
    monkeypatch.setattr(omni, "usd", module, raising=False)
    # Reversed body order detects accidentally using traversal order as the body index.
    env = SimpleNamespace(
        num_envs=2, device="cuda:0", body_names=["gripperMover", "link06"],
        sim=SimpleNamespace(stage=stage),
    )
    factory = RobotOcclusionGeometryBatch(env)
    assert {index for index, _ in factory.body_meshes} == {0, 1}
    states = torch.zeros(2, 2, 13, device="cuda")
    states[..., 3] = 1
    states[:, 0, 0] = 0.2
    states[1, :, 1] = 2
    starts, ends = [], []
    for index, mesh in factory.body_meshes:
        triangle = mesh.points[mesh.faces[:3].long()]
        center = triangle.mean(0)
        normal = torch.cross(triangle[1] - triangle[0], triangle[2] - triangle[0], dim=-1)
        normal = normal / normal.norm()
        starts.append(center[None] + states[:, index, :3] + normal * 0.05)
        ends.append(center[None] + states[:, index, :3] - normal * 0.05)
    starts, ends = torch.stack(starts, 1), torch.stack(ends, 1)
    geometry = factory.update(states)
    assert not geometry.visible(starts, ends).any()
    for row in range(2):
        reference = merged_geometry(factory.body_meshes, states[row])
        torch.testing.assert_close(
            geometry.visible(starts, ends)[row], reference.visible(starts[row], ends[row])
        )
