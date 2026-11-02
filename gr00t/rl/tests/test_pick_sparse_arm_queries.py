"""Sparse collision requests preserve owner frames and omit inactive FK work."""

import pytest
import torch

from gr00t.rl.pick.collision import ArmCollisionSamples


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sparse_arm_queries_keep_dense_distances_and_batch_owners(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU collision requests")
    arm = ArmCollisionSamples.__new__(ArmCollisionSamples)
    arm.parts = [(0, torch.zeros(2, 3, device=device)), (2, torch.ones(3, 3, device=device))]
    root = torch.zeros(3, 2, 4, 13, device=device)
    root[..., 3] = 1
    root[..., 0] = torch.arange(3, device=device)[:, None, None]
    joints = torch.full((3, 2, 4, 6), 0.1, device=device)
    counts = []

    class Geometry:
        def __init__(self, labels):
            self.labels = labels

        def select(self, rows):
            return Geometry(self.labels[rows])

        def rigid_surface_clearance(self, positions, orientations, points, **kwargs):
            counts.append(positions.numel() // 3)
            labels = self.labels.reshape(-1, *([1] * (positions.ndim - 2)))
            return positions[..., 0] * 0.001 + labels * 0.01 + 0.1

    geometry = Geometry(torch.arange(3, device=device).float())
    dense = arm.clearance(geometry, root, joints)
    mask = torch.arange(24, device=device).reshape(3, 2, 4).remainder(3) == 0
    counts.clear()

    def forbidden(*args, **kwargs):
        raise AssertionError("Sparse collision requests must not transfer tensors to the host")

    with monkeypatch.context() as scope:
        for name in ("cpu", "item", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        sparse = arm.clearance(geometry, root, joints, query_mask=mask)
    torch.testing.assert_close(sparse[mask], dense[mask], rtol=0, atol=0)
    assert sparse[~mask].eq(2).all()
    assert counts == [8, 8]
    counts.clear()
    empty = arm.clearance(geometry, root, joints, query_mask=torch.zeros_like(mask))
    assert not counts and empty.eq(2).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("batched", [False, True])
def test_grasp_environment_queries_pack_only_true_mask_entries(monkeypatch, device, batched):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU collision requests")
    from gr00t.rl.pick.planning import grasp_environment_clearance

    positions = torch.arange(72, device=device).reshape(3, 8, 3).float() * 0.001
    q = positions.new_tensor([1, 0, 0, 0]).expand(3, 8, -1)
    aperture = positions.new_full((3, 8), 0.2)
    calls = []

    class Geometry:
        supports_query_mask = True

        def __init__(self, labels):
            self.labels = labels

        def select(self, rows):
            return Geometry(self.labels[rows])

    class Collision:
        def gripper_clearance(self, geometry, p, orientations, **kwargs):
            assert "query_mask" not in kwargs
            calls.append(p.numel() // 3)
            return p[..., 0] + 0.1 + (geometry.labels[:, None] * 0.001 if batched else 0)

        def gripper_closing_clearance(self, geometry, p, orientations, angles, **kwargs):
            return self.gripper_clearance(geometry, p, orientations, **kwargs) + angles

    geometry = Geometry(torch.arange(3, device=device)) if batched else None
    collision = Collision()
    dense = grasp_environment_clearance(collision, geometry, positions, q, aperture)
    calls.clear()
    mask = torch.arange(24, device=device).reshape(3, 8).remainder(3) == 0

    def forbidden(*args, **kwargs):
        raise AssertionError("Grasp collision request packing must remain on the GPU")

    with monkeypatch.context() as scope:
        for name in ("cpu", "item", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        sparse = grasp_environment_clearance(
            collision, geometry, positions, q, aperture, query_mask=mask
        )
    torch.testing.assert_close(sparse[mask], dense[mask], rtol=0, atol=0)
    assert sparse[~mask].isnan().all()
    assert calls == [8, 8]
    calls.clear()
    empty = grasp_environment_clearance(
        collision, geometry, positions, q, aperture, query_mask=torch.zeros_like(mask)
    )
    assert not calls and empty.isnan().all()
