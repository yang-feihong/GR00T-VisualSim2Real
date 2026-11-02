"""Held payload candidates and geometry remain independent across basket scenes."""

from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.auto_open.motion_math import _euler_xyz_quat, _quat_multiply
from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.pick import basket_carry
from gr00t.rl.pick.basket import BasketGeometry
from gr00t.rl.pick.basket_carry import (
    held_geometry,
    held_geometry_batch,
    held_pose_bank,
    held_pose_bank_batch,
)
from gr00t.rl.pick.batched_geometry import BatchedSceneGeometry
from gr00t.rl.pick.geometry_views import TranslatedGeometry
from gr00t.rl.pick.scene_geometry import SceneGeometry

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU basket geometry")


def cases():
    root = torch.zeros(2, 13, device="cuda")
    root[:, :3] = root.new_tensor([[0, 0, 0.52], [10, 0, 0.47]])
    root[:, 3:7] = _euler_xyz_quat(root.new_tensor([[0, 0, 0], [0.12, 0.1, 1.0]]))
    obj = root.clone()
    obj[:, :3] = root[:, :3] + _quat_rotate(root[:, 3:7], root.new_tensor([[-0.2, 0, 0.16]]))
    ee = obj[:, :3] + _quat_rotate(root[:, 3:7], root.new_tensor([[0.1, 0, 0.1]]))
    ee_q = _quat_multiply(root[:, 3:7], _euler_xyz_quat(root.new_tensor([[0, 1.57, 0]])))
    state = {"root_states": root, "ee_pos": ee, "ee_orn": ee_q, "object_state": obj}
    signs = torch.cartesian_prod(*[root.new_tensor([-1, 1])] * 3)
    points = root.new_full((2, 12, 3), torch.nan)
    points[:, :8] = signs[None] * root.new_tensor([[[0.02, 0.02, 0.02]], [[0.025, 0.015, 0.02]]])
    points[1, 8:] = points[1, :4] * 0.8
    mask = torch.arange(12, device="cuda")[None] < torch.tensor([8, 12], device="cuda")[:, None]
    basket = BasketGeometry.__new__(BasketGeometry)
    basket.boxes = (signs * root.new_tensor([0.3, 0.3, 0.1]) + root.new_tensor([-0.2, 0, 0.1]))[
        None
    ]
    basket.faces = torch.tensor([[0, 1, 2]], device="cuda")
    basket.floor_lower = root.new_tensor([-0.5, -0.3, 0.1])
    basket.floor_upper = root.new_tensor([0.1, 0.3, 0.1])
    basket.rim_z = root.new_tensor(0.2)
    return state, points, mask, basket


def test_held_candidate_bank_preserves_each_roots_rim_and_payload_radius(monkeypatch):
    state, points, mask, basket = cases()
    heights = state["root_states"][:, 2] + 0.5

    def forbidden(*args, **kwargs):
        raise AssertionError("Held candidate generation must remain on the GPU")

    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        together = held_pose_bank_batch(state, basket, points, heights, support_mask=mask)
    for row in range(2):
        split = held_pose_bank(
            {key: value[row : row + 1] for key, value in state.items()},
            basket,
            points[row, mask[row]],
            heights[row : row + 1],
        )
        for batched, single in zip(together[:3], split[:3]):
            torch.testing.assert_close(batched[row], single)
        torch.testing.assert_close(together[3], split[3])
    torch.testing.assert_close(
        together[0][:, :, -1, 2], (heights + 0.005)[:, None].expand(-1, len(together[3]))
    )


class Geometry:
    def __init__(self, source):
        self.nonrobot_geometry = source

    def select(self, rows):
        return Geometry(self.nonrobot_geometry.select(rows))


def test_held_geometry_matches_single_scene_queries_with_padded_hulls_and_memory_chunks(
    monkeypatch,
):
    state, points, mask, basket = cases()
    centers, quats, relative, _ = held_pose_bank_batch(
        state, basket, points, state["root_states"][:, 2] + 0.5, support_mask=mask
    )
    # Several retract choices and the first transverse tilt; all object samples remain present.
    centers, quats = centers[:, :8], quats[:, :8]
    positions = centers - _quat_rotate(quats, relative[:, None, None])
    cloud = points.new_tensor([[-3, -3, -1], [3, -3, -1], [3, 3, -1], [-3, 3, -1]])
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], device="cuda")
    scene = SceneGeometry(cloud, faces)
    offsets = state["root_states"][:, :3] * points.new_tensor([1, 0, 0])
    geometry = Geometry(
        BatchedSceneGeometry([scene], torch.zeros(2, device="cuda", dtype=torch.long), offsets)
    )
    collision = SimpleNamespace(
        gripper_pose_clearance=lambda g, p, q, a, **kw: p.new_full(p.shape[:-1], 0.02)
    )
    aperture = points.new_full((2,), 0.5)
    arguments = (state, positions, quats, relative, points, basket, collision, geometry, aperture)

    def forbidden(*args, **kwargs):
        raise AssertionError("Held surface certification must remain on the GPU")

    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        together = held_geometry_batch(*arguments, support_mask=mask)
    for row in range(2):
        local = {key: value[row : row + 1] for key, value in state.items()}
        view = SimpleNamespace(nonrobot_geometry=TranslatedGeometry(scene, offsets[row]))
        split = held_geometry(
            local,
            positions[row],
            quats[row],
            relative[row],
            points[row, mask[row]],
            basket,
            collision,
            view,
            aperture[row : row + 1],
        )
        for combined, single in zip(together, split):
            torch.testing.assert_close(combined[row], single)
    monkeypatch.setattr(
        basket_carry, "MAX_HELD_SAMPLE_EVALUATIONS", positions.shape[2] * points.shape[1]
    )
    chunked = held_geometry_batch(*arguments, support_mask=mask)
    for expected, actual in zip(together, chunked):
        torch.testing.assert_close(actual, expected)
