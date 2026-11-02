"""Stable-face screening is a generic offline asset audit, not a seed-specific grasp."""

import math

import numpy as np
import pytest
import torch

from gr00t.rl.auto_open.motion_math import _euler_xyz_quat, _quat_multiply
from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.pick.grasp_quality import convex_grasp_quality
from gr00t.rl.pick.planning import object_frame_grasp_bank
from gr00t.rl.pick.pocket import BASKET_MINIMUM_CENTER_DEPTH_M
from gr00t.rl.pick.supported_asset_audit import (
    basket_supported_shapes,
    gripper_collision_clouds,
    stable_pose_bank,
    support_plane_clearance,
)


def test_basket_publication_does_not_admit_generic_top_down_only_shapes():
    def pose(generic, basket):
        return {
            "supported_grasp_count": generic,
            "top_down_grasp_count": generic,
            "basket_top_down_grasp_count": basket,
        }

    rejected, accepted = pose(99, 0), pose(100, 3)
    shapes = [
        {"scale_xyz": [1, 1, 1], "poses": [rejected]},
        {"scale_xyz": [0.8, 0.9, 1.0], "poses": [rejected, accepted]},
    ]
    result = basket_supported_shapes(shapes)
    assert len(result) == 1
    assert result[0]["scale_xyz"] == [0.8, 0.9, 1.0]
    assert len(result[0]["poses"]) == 1
    assert result[0]["poses"][0]["supported_grasp_count"] == 3
    assert result[0]["poses"][0]["top_down_grasp_count"] == 3
    assert accepted["supported_grasp_count"] == 100


def test_object_frame_candidates_rotate_with_the_object_not_with_world_axes():
    identity = torch.tensor([[1.0, 0, 0, 0]])
    rotation = _euler_xyz_quat(torch.tensor([[0.1, -0.2, 0.7]]))
    local = object_frame_grasp_bank(identity, [0.05, 0.06, 0.08])
    world = object_frame_grasp_bank(rotation, [0.05, 0.06, 0.08])
    torch.testing.assert_close(world, _quat_multiply(rotation[:, None], local))
    torch.testing.assert_close(world.norm(dim=-1), torch.ones_like(world[..., 0]))


def test_stable_face_bank_is_seed_independent_and_uses_real_resting_faces():
    import trimesh

    mesh = trimesh.creation.box(extents=[0.05, 0.06, 0.08])
    first, probabilities = stable_pose_bank(mesh.vertices, mesh.faces)
    np.random.seed(99)
    np.random.random(100)
    second, repeated = stable_pose_bank(mesh.vertices, mesh.faces)
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(probabilities, repeated)
    assert len(first) == 6
    np.testing.assert_allclose(np.linalg.det(first), 1, atol=1e-8)
    np.testing.assert_allclose(np.abs(first[:, 2]).max(-1), 1, atol=1e-8)
    with pytest.raises(ValueError, match="At least one"):
        stable_pose_bank(mesh.vertices, mesh.faces, max_poses=0)


def test_support_plane_projection_matches_full_jaw_vertex_sweep():
    tool = _euler_xyz_quat(torch.tensor([[0.4, 1.1, -0.3]]))[:, None]
    obj = torch.tensor([[1.0, 0, 0, 0]])
    points = torch.cartesian_prod(*[torch.tensor([-0.025, 0.025])] * 3)
    center = torch.tensor([[[[0.08, 0.0, 0.025]]]])
    aperture = torch.tensor([[[0.4]]])
    actual = support_plane_clearance(tool, obj, points)(center, aperture)
    fixed, moving = gripper_collision_clouds("cpu")
    origin = points.new_tensor([0.0, 0, 0.025]) - _quat_rotate(tool[0, 0], center[0, 0, 0])
    minima = [(_quat_rotate(tool[0, 0], fixed) + origin)[:, 2].min()]
    for angle in torch.linspace(0.4, math.pi / 3, 7):
        opened = torch.stack(
            (
                angle.cos() * moving[:, 0] - angle.sin() * moving[:, 2],
                moving[:, 1],
                angle.sin() * moving[:, 0] + angle.cos() * moving[:, 2],
            ),
            -1,
        )
        minima.append((_quat_rotate(tool[0, 0], opened) + origin)[:, 2].min())
    torch.testing.assert_close(actual.reshape(()), torch.stack(minima).min(), atol=1e-6, rtol=0)
    offset = support_plane_clearance(tool, obj, points, command_seating_offset_m=0.0005)(
        center, aperture
    )
    normal_z = _quat_rotate(tool[0, 0], points.new_tensor([0, 0, 1]))[2]
    torch.testing.assert_close(offset - actual, torch.ones_like(actual) * 0.0005 * normal_z)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batched_support_planes_match_each_scalar_with_padding_and_reordering(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU support planes")
    cube = torch.cartesian_prod(*[torch.tensor([-0.025, 0.025], device=device)] * 3)
    points = torch.full((2, 10, 3), torch.nan, device=device)
    points[:, :8] = cube
    points[1, :8, 2] += 0.01
    mask = torch.arange(10, device=device)[None].expand(2, -1) < 8
    tool = _euler_xyz_quat(
        torch.tensor(
            [[0.4, 1.1, -0.3], [0, 1.3, 0.2], [0.2, 1.4, 0.5], [0.1, 1.2, -0.2]], device=device
        )
    ).reshape(2, 2, 4)
    obj = _euler_xyz_quat(torch.tensor([[0, 0, 0], [0.1, -0.1, 0.3]], device=device))
    center = cube.new_tensor([0.08, 0, 0.025]).expand(2, 2, 2, 3)
    aperture = cube.new_tensor([0.2, 0.4]).expand(2, 2, 2)
    gripper_collision_clouds(str(points.device))

    def forbidden(*args, **kwargs):
        raise AssertionError("Batched support planes must not read GPU values on the host")

    with monkeypatch.context() as scope:
        for name in ("cpu", "item", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        together = support_plane_clearance(tool, obj, points, support_mask=mask)(center, aperture)
        reordered = support_plane_clearance(
            tool.flip(0), obj.flip(0), points.flip(0), support_mask=mask.flip(0)
        )(center, aperture)
    torch.testing.assert_close(reordered.flip(0), together)
    for row in range(2):
        scalar = support_plane_clearance(
            tool[row : row + 1], obj[row : row + 1], points[row, mask[row]]
        )(center[row : row + 1], aperture[row : row + 1])
        torch.testing.assert_close(scalar, together[row : row + 1])
    with pytest.raises(ValueError, match="Support mask"):
        support_plane_clearance(tool, obj, points, support_mask=mask[:, :3])


def test_basket_capture_depth_rejects_shallow_centers_without_changing_geometry():
    if not torch.cuda.is_available():
        pytest.skip("Actual jaw contact profiles require CUDA")
    identity = torch.tensor([[1.0, 0, 0, 0]], device="cuda")
    tool = object_frame_grasp_bank(identity, [0.04, 0.05, 0.06])[:, :8]
    support = torch.cartesian_prod(
        identity.new_tensor([-0.02, 0.02]),
        identity.new_tensor([-0.025, 0.025]),
        identity.new_tensor([-0.03, 0.03]),
    )
    original = convex_grasp_quality(tool, identity, support, return_depth_candidates=True)
    robust = convex_grasp_quality(
        tool,
        identity,
        support,
        return_depth_candidates=True,
        minimum_center_depth_m=BASKET_MINIMUM_CENTER_DEPTH_M,
    )
    depth_ok = original["center_depth_margin_m"] >= BASKET_MINIMUM_CENTER_DEPTH_M
    assert bool((original["pocket_fit"] & ~depth_ok).any())
    assert bool(robust["pocket_fit"].any())
    torch.testing.assert_close(robust["pocket_fit"], original["pocket_fit"] & depth_ok)
    for key in ("center_local", "aperture_angle_rad", "center_depth_margin_m"):
        torch.testing.assert_close(robust[key], original[key])


@pytest.mark.parametrize("depth", [0, -0.01, float("nan"), float("inf")])
def test_invalid_center_depth_cannot_weaken_capture_gate(depth):
    with pytest.raises(ValueError, match="Minimum center depth"):
        convex_grasp_quality(None, None, None, minimum_center_depth_m=depth)
