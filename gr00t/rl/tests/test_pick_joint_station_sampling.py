import pytest
import torch
from gr00t.rl.pick.station_sampling import joint_station_requests

from gr00t.rl.pick import station_sampling


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_joint_budget_masking_and_split_invariance(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pocket = torch.ones(4, 400, device=device, dtype=torch.bool)
    corridor = torch.ones(4, 27, device=device, dtype=torch.bool)
    pocket[0, :30] = False
    pocket[1] = False
    corridor[2, :12] = False
    seeds = torch.tensor([1700, 1701, 1702, 1703], device=device)
    owner, body, candidate = joint_station_requests(pocket, corridor, 16, 15, seeds)
    assert torch.bincount(owner, minlength=4).max() <= 1000
    assert not (owner == 1).any()
    assert pocket[owner, candidate].all()
    assert corridor[owner, body // (16 * 15)].all()
    for row in (0, 2, 3):
        one = joint_station_requests(
            pocket[row : row + 1], corridor[row : row + 1], 16, 15, seeds[row : row + 1]
        )
        torch.testing.assert_close(one[1], body[owner == row])
        torch.testing.assert_close(one[2], candidate[owner == row])


@pytest.mark.parametrize("device", ["cuda"])
def test_direction_depth_sample_precedes_contact_and_is_batch_independent(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    calls = []

    original = station_sampling.convex_grasp_quality

    def quality(q, object_q, support, **kwargs):
        assert support.shape[1] <= 10000
        calls.append((q.shape[1], support.shape[1]))
        return original(q, object_q, support, **kwargs)

    class Geometry:
        def select(self, rows):
            return self

    class Collision:
        def gripper_clearance(self, geometry, p, q, **kwargs):
            return p.new_full(p.shape[:-1], 0.01)

        def gripper_closing_clearance(self, geometry, p, q, angle, **kwargs):
            return p.new_full(p.shape[:-1], 0.01)

    monkeypatch.setattr(station_sampling, "convex_grasp_quality", quality)
    monkeypatch.setattr(
        station_sampling,
        "gripper_collision_clouds",
        lambda _: (torch.zeros(3, 3, device=device),) * 2,
    )
    q = torch.zeros(3, 96, 4, device=device)
    angle = torch.arange(96, device=device) / 96
    q[..., 0], q[..., 3] = angle.cos(), angle.sin()
    obj = torch.zeros(3, 13, device=device)
    obj[:, 3] = 1
    state = {"object_state": obj, "episode_seed": torch.tensor([1, 20, 40], device=device)}
    axis = torch.arange(96, device=device, dtype=torch.float32)
    z = 1 - 2 * (axis + 0.5) / 96
    phi = axis * 2.399963
    radius = (1 - z.square()).sqrt()
    support = 0.02 * torch.stack((radius * phi.cos(), radius * phi.sin(), z), -1)
    support = support[None].expand(3, -1, -1)
    mask = torch.ones(support.shape[:2], device=device, dtype=torch.bool)
    result = station_sampling.sample_station_pockets(
        q, state, support, mask, Collision(), Geometry(), count=8, object_pose_offset=64
    )
    for row in range(3):
        one = station_sampling.sample_station_pockets(
            q[row : row + 1],
            {key: value[row : row + 1] for key, value in state.items()},
            support[row : row + 1],
            mask[row : row + 1],
            Collision(),
            Geometry(),
            count=8,
            object_pose_offset=64,
        )
        torch.testing.assert_close(one[0], result[0][row : row + 1])
        torch.testing.assert_close(one[1], result[1][row : row + 1], equal_nan=True)
        for key in one[2]:
            torch.testing.assert_close(one[2][key], result[2][key][row : row + 1], equal_nan=True)
    assert [size for size, points in calls if points == 1024] == [32] * 12
    assert [size for size, points in calls if points == 9984] == [8] * 4
