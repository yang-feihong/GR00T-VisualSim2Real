"""Seated carry paths preserve per-environment durations, scenes and validity."""

from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.pick import planning
from gr00t.rl.pick.batched_geometry import BatchedSceneGeometry
from gr00t.rl.pick.scene_geometry import SceneGeometry

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU carry planning")


def carry_case(monkeypatch, *, reject_second=False):
    device = "cuda"
    roots = torch.zeros(2, 13, device=device)
    roots[:, 3] = 1
    roots[:, 2] = torch.tensor([0.36, 0.5], device=device)
    roots[1, 0] = 20
    state = {
        "root_states": roots,
        "dof_pos": torch.zeros(2, 6, device=device),
        "ee_pos": torch.tensor([[0.5, 0, 0.11], [20.5, 0, 0.65]], device=device),
        "ee_orn": torch.tensor([[1, 0, 0, 0]], device=device, dtype=torch.float32).expand(2, -1),
        "object_state": torch.tensor([[0.5, 0, 0.0244], [20.5, 0, 0.55]], device=device),
    }
    points = torch.tensor([[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]], device=device)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], device=device)
    scenes = [
        SceneGeometry(points, faces),
        SceneGeometry(points + points.new_tensor([0, 0, 0.5]), faces),
    ]
    geometry = BatchedSceneGeometry(
        scenes, torch.tensor([0, 1], device=device), roots[:, :3] * roots.new_tensor([1, 0, 0])
    )
    calls = []

    def inverse(p, q, roots, seeds, limits):
        calls.append(p.shape)
        exact = torch.ones(p.shape[:2], dtype=torch.bool, device=device)
        if reject_second:
            exact &= roots[:, :1] < 10
        return SimpleNamespace(
            exact=exact, arm_joint_positions=torch.zeros(*p.shape[:2], 6, device=device)
        )

    monkeypatch.setattr(planning, "inverse_bank", inverse)

    def gripper(geometry, p, q, aperture, **kwargs):
        return geometry.rigid_surface_clearance(p, q, points.new_zeros(1, 3), **kwargs)

    collision = SimpleNamespace(
        arm_joint_ids=torch.arange(6, device=device),
        clearance=lambda geometry, roots, joints, **kw: torch.full(
            roots.shape[:-1], 0.01, device=device
        ),
        gripper_pose_clearance=gripper,
    )
    return state, geometry, collision, calls


@pytest.mark.parametrize("reject_second", [False, True])
def test_batch_carry_matches_split_calls_without_host_scalar_reads(monkeypatch, reject_second):
    state, geometry, collision, calls = carry_case(monkeypatch, reject_second=reject_second)
    heights = state["object_state"][:, 2].clone()
    displacement = torch.zeros(2, 3, device="cuda")
    displacement[1, 0] = -0.08
    aperture = torch.full((2,), 0.6, device="cuda")
    limits = torch.tensor([[-3.0, 3.0]] * 6, device="cuda")

    def no_host_read(*args, **kwargs):
        raise AssertionError("Planning must not read a GPU tensor on the CPU")

    with monkeypatch.context() as scope:
        scope.setattr(torch.Tensor, "item", no_host_read)
        scope.setattr(torch.Tensor, "cpu", no_host_read)
        path, valid, diagnostics = planning.plan_carry_batch(
            state, displacement, heights, aperture, limits, collision, geometry, 0.02, capacity=512
        )
    assert calls == [(2 * (len(planning.GRASP_BODY_HEIGHTS_M) + 3) * 3, 17, 3)]
    assert valid.tolist() == [True, not reject_second]
    assert path[2][0] != path[2][1]
    assert path[0].device.type == "cuda"
    assert (path[0].diff(dim=1).norm(dim=-1) / 0.02).max() <= 0.15 + 1e-5
    for row in range(2):
        single = planning.plan_carry_batch(
            {key: value[row : row + 1] for key, value in state.items()},
            displacement[row : row + 1],
            heights[row : row + 1],
            aperture[row : row + 1],
            limits,
            collision,
            geometry.select(torch.tensor([row], device="cuda")),
            0.02,
            capacity=512,
        )
        for actual, expected in zip(path, single[0]):
            torch.testing.assert_close(actual[row : row + 1], expected)
        torch.testing.assert_close(valid[row : row + 1], single[1])
        for key, value in diagnostics.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(value[row : row + 1], single[2][key])
    _, capacity_valid, diagnostics = planning.plan_carry_batch(
        state, displacement, heights, aperture, limits, collision, geometry, 0.02, capacity=2
    )
    assert not capacity_valid.any()
    assert not diagnostics["within_path_capacity"].any()


def test_native_inverse_carry_admission_is_independent_of_batch_layout(monkeypatch):
    native_inverse = planning.inverse_bank
    state, geometry, collision, _ = carry_case(monkeypatch)
    monkeypatch.setattr(planning, "inverse_bank", native_inverse)
    displacement = state["ee_pos"].new_zeros(2, 3)
    heights = state["object_state"][:, 2].clone()
    aperture = state["ee_pos"].new_full((2,), 0.6)
    limits = state["ee_pos"].new_tensor([[-3.0, 3.0]] * 6)
    result = planning.plan_carry_batch(
        state, displacement, heights, aperture, limits, collision, geometry, 0.02, capacity=512
    )
    for row in range(2):
        independent = planning.plan_carry_batch(
            {key: value[row : row + 1] for key, value in state.items()},
            displacement[row : row + 1],
            heights[row : row + 1],
            aperture[row : row + 1],
            limits,
            collision,
            geometry.select(torch.tensor([row], device="cuda")),
            0.02,
            capacity=512,
        )
        torch.testing.assert_close(result[1][row : row + 1], independent[1])
        torch.testing.assert_close(
            result[2]["safe_inverse_sample_mask"][row : row + 1],
            independent[2]["safe_inverse_sample_mask"],
        )
