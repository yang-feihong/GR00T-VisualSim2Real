"""Physical basket admission checks use actual support BVHs and masked object hulls."""

from pathlib import Path

import numpy as np
import pytest
import torch

from gr00t.rl.auto_open.motion_math import _euler_xyz_quat
from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.pick.basket import BasketGeometry
from gr00t.rl.pick.basket_initialization import (
    basket_object_checks_batch,
    load_basket_payload_hulls,
)


@pytest.fixture(scope="module")
def basket_model():
    if not torch.cuda.is_available():
        pytest.skip("Basket support uses CUDA spatial queries")
    return BasketGeometry.from_robot_usd()


def _single_support_reference(basket, root, poses, hulls):
    inverse = root[3:7].clone()
    inverse[1:] *= -1
    world = _quat_rotate(poses[:, None, 3:7], hulls) + poses[:, None, :3]
    local = _quat_rotate(inverse, world - root[:3])
    lower, upper = local.amin(1), local.amax(1)
    margin = torch.minimum(
        lower[:, :2] - basket.floor_lower[:2], basket.floor_upper[:2] - upper[:, :2]
    ).amin(-1)
    center = _quat_rotate(inverse, poses[:, :3] - root[:3])
    height, _ = basket.support(center[:, :2])
    error = lower[:, 2] - height
    return {
        "all_objects_inside_basket": (margin >= 0).all()[None],
        "all_objects_supported": (error.abs() < 0.02).all()[None],
    }, {"object_wall_margin_m": margin, "object_support_error_m": error}


def test_payload_hull_metadata_loads_unique_assets_once_and_preserves_absent_slots(monkeypatch):
    data = {
        "target": np.zeros((8, 3), dtype=np.float32),
        "prop": np.zeros((5, 3), dtype=np.float32),
    }
    loaded = []

    def read(path):
        loaded.append(Path(path).parent.name)
        return data[Path(path).parent.name]

    monkeypatch.setattr(np, "load", read)
    episodes = [
        {
            "asset": {"usd_path": "target/object.usd"},
            "distractors": [{"asset": {"usd_path": "prop/object.usd"}}],
        },
        {"asset": {"usd_path": "target/object.usd"}, "distractors": []},
    ]
    hulls, points, payloads = load_basket_payload_hulls(episodes, 2, "cpu")
    assert loaded == ["target", "prop"]
    assert hulls.shape == (2, 2, 8, 3)
    assert points.sum(-1).tolist() == [[8, 5], [8, 0]]
    assert payloads.tolist() == [[True, True], [True, False]]
    assert hulls[1, 1].isnan().all()
    with pytest.raises(ValueError, match="capacity omits"):
        load_basket_payload_hulls(episodes, 1, "cpu")
    assert loaded == ["target", "prop"]


def test_basket_support_batch_matches_scalar_with_rotated_trunks_and_absent_slots(
    basket_model, monkeypatch
):
    basket = basket_model
    reference = basket.boxes
    roots = reference.new_zeros(2, 13)
    roots[:, :3] = reference.new_tensor([[0, 0, 0.5], [10, 0, 0.6]])
    roots[:, 3:7] = _euler_xyz_quat(reference.new_tensor([[0.1, -0.1, 0.3], [-0.1, 0.1, 1.2]]))
    signs = torch.cartesian_prod(*[reference.new_tensor([-1, 1])] * 3)
    hull = signs[None] * reference.new_tensor([[[0.02, 0.02, 0.02]], [[0.025, 0.015, 0.03]]])
    points = reference.new_full((2, 2, 12, 3), torch.nan)
    points[:, :, :8] = hull[None]
    point_mask = torch.arange(12, device=reference.device)[None, None].expand(2, 2, -1) < 8
    payload_mask = torch.tensor([[True, True], [True, False]], device=reference.device)
    center = (basket.floor_lower[:2] + basket.floor_upper[:2]) / 2
    local = reference.new_zeros(2, 2, 3)
    local[..., :2] = center + reference.new_tensor([[[-0.035, 0], [0.035, 0.02]]])
    height, _ = basket.support(local[..., :2])
    local[..., 2] = height + hull[..., 2].abs().amax(-1)[None]
    local[0, 1, 2] += 0.03
    poses = roots[:, None].expand(-1, 2, -1).clone()
    poses[..., :3] = roots[:, None, :3] + _quat_rotate(roots[:, None, 3:7], local)
    poses[1, 1] = torch.nan
    state = {"root_states": roots, "object_state": poses[:, 0], "distractor_states": poses[:, 1:]}

    def forbidden(*args, **kwargs):
        raise AssertionError("Basket support checks must not convert tensor values to host scalars")

    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist", "__float__", "__int__"):
            scope.setattr(torch.Tensor, name, forbidden)
        checks, metrics = basket_object_checks_batch(
            basket, state, points, point_mask, payload_mask
        )
    assert checks["all_objects_inside_basket"].all()
    assert checks["all_objects_supported"].tolist() == [False, True]
    torch.testing.assert_close(
        metrics["object_support_error_m"][0], reference.new_tensor([0, 0.03]), atol=1e-5, rtol=0
    )
    for row in range(2):
        scalar, detail = _single_support_reference(
            basket, roots[row], poses[row, payload_mask[row]], hull[payload_mask[row]]
        )
        for key in scalar:
            torch.testing.assert_close(checks[key][row : row + 1], scalar[key])
        for key in metrics:
            torch.testing.assert_close(
                metrics[key][row, payload_mask[row]],
                detail[key],
                atol=1e-5,
                rtol=1e-5,
            )
    reverse = torch.tensor([1, 0], device=reference.device)
    reordered, _ = basket_object_checks_batch(
        basket,
        {key: value[reverse] for key, value in state.items()},
        points[reverse],
        point_mask[reverse],
        payload_mask[reverse],
    )
    for key in checks:
        torch.testing.assert_close(reordered[key][reverse], checks[key])
