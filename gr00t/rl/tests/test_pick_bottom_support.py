"""Physical contact footprint is independent of the object's upper overhang."""

import pytest
import torch

from gr00t.rl.pick.support_features import (
    asset_support_features,
    bottom_contact_extrema,
    supported_contact_margin,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_asset_features_are_batched_cached_and_keep_true_centroid(tmp_path, monkeypatch, device):
    from types import SimpleNamespace

    import numpy as np

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    points = np.asarray(
        [[x, y, z] for x in (-0.02, 0.02) for y in (-0.01, 0.01) for z in (-0.03, 0.03)],
        dtype=np.float32,
    ) + np.asarray([0.003, 0.001, 0.005], dtype=np.float32)
    faces = np.asarray(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 6, 7],
            [4, 7, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ]
    )
    np.save(tmp_path / "collision_support.npy", points)
    np.save(tmp_path / "collision_faces.npy", faces)
    asset = {"usd_path": str(tmp_path / "object.usda"), "size_m": [0.04, 0.02, 0.06]}
    box = {"size_m": [0.03, 0.03, 0.03]}
    geometry = SimpleNamespace(device=device)

    def forbidden(*args, **kwargs):
        raise AssertionError("Asset feature reductions must not transfer tensor data to the host")

    with monkeypatch.context() as scope:
        for name in ("cpu", "item", "tolist", "numpy"):
            scope.setattr(torch.Tensor, name, forbidden)
        contact, center = asset_support_features([asset, box, asset], geometry)
        reordered = asset_support_features([box, asset], geometry)
    torch.testing.assert_close(center[0], center.new_tensor([0.003, 0.001, 0.005]))
    torch.testing.assert_close(center[1], center.new_zeros(3))
    torch.testing.assert_close(contact[0], contact[2])
    torch.testing.assert_close(reordered[0], contact[[1, 0]])
    torch.testing.assert_close(reordered[1], center[[1, 0]])
    assert len(geometry.pick_asset_support_cache) == 2


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_contact_polygon_rejects_missing_support_without_using_upper_overhang(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    points = torch.tensor(
        [
            [
                [-0.01, -0.01, 0],
                [0.01, -0.01, 0],
                [0.01, 0.01, 0],
                [-0.01, 0.01, 0],
                [0.08, 0.08, 0.02],
            ],
        ],
        device=device,
    )
    contacts = bottom_contact_extrema(
        points, torch.ones(1, 5, device=device, dtype=torch.bool), contact_band_m=0.002
    )
    assert (contacts[..., :2].abs() <= 0.01).all()
    mask = torch.ones(1, 8, device=device, dtype=torch.bool)
    center = points.new_zeros(1, 3)
    torch.testing.assert_close(
        supported_contact_margin(contacts, mask, center), points.new_tensor([0.01])
    )
    mask &= contacts[..., 0] > 0
    assert (supported_contact_margin(contacts, mask, center) < 0).all()
    torch.testing.assert_close(
        supported_contact_margin(contacts + 3, torch.ones_like(mask), center + 3),
        points.new_tensor([0.01]),
    )
