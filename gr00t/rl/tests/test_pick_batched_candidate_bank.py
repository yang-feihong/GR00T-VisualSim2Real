"""The production grasp candidate generator keeps heterogeneous GPU rows isolated."""

from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.auto_open.motion_math import _euler_xyz_quat
from gr00t.rl.pick import planning

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU grasp geometry")


def cases():
    root = torch.zeros(2, 13, device="cuda")
    root[:, 3], root[:, 2] = 1, 0.52
    root[1, 0] = 32
    q = _euler_xyz_quat(torch.tensor([[0.0, 1.57, 0], [0.0, 0, 1.57]], device="cuda"))
    target = torch.tensor([[0.6, 0, 0.03], [32.6, 0.1, 0.72]], device="cuda")
    state = {"root_states": root, "ee_orn": q, "object_state": root.clone()}
    state["object_state"][:, :3] = target
    sizes = torch.tensor([[0.04, 0.04, 0.04], [0.06, 0.04, 0.05]], device="cuda")
    corners = torch.cartesian_prod(*[torch.tensor([-1.0, 1], device="cuda")] * 3)
    hulls = torch.full((2, 12, 3), torch.nan, device="cuda")
    hulls[:, :8] = corners[None] * sizes[:, None] / 2
    hulls[1, 8:] = hulls[1, :4] * 0.8
    mask = torch.arange(12, device="cuda")[None] < torch.tensor([8, 12], device="cuda")[:, None]
    return state, target, sizes, hulls, mask


@pytest.mark.parametrize("retain_depth", [False, True])
def test_native_pocket_bank_matches_split_for_different_hulls_and_depth_preferences(
    retain_depth, monkeypatch
):
    state, target, sizes, hulls, mask = cases()
    orientations = torch.stack((state["ee_orn"], state["object_state"][:, 3:7]), 1)
    preference = target.new_tensor([0.0, 0.8])

    def clearance(geometry, p, q, *args, **kwargs):
        return p.new_full(p.shape[:2], 0.02)

    collision = SimpleNamespace(gripper_clearance=clearance, gripper_closing_clearance=clearance)
    geometry = SimpleNamespace(supports_query_mask=True)
    options = dict(
        candidate_orientations=orientations,
        pocket_depth_preference=preference,
        retain_depth_candidates=retain_depth,
        support_mask=mask,
    )
    result = planning.grasp_candidate_bank(
        state, target, sizes, collision, geometry, hulls, **options
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Native pocket evaluation must remain on the GPU after asset loading")

    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        planning.grasp_candidate_bank(state, target, sizes, collision, geometry, hulls, **options)
    for row in range(2):
        single = planning.grasp_candidate_bank(
            {key: value[row : row + 1] for key, value in state.items()},
            target[row : row + 1],
            sizes[row : row + 1],
            collision,
            geometry,
            hulls[row : row + 1],
            candidate_orientations=orientations[row : row + 1],
            pocket_depth_preference=preference[row : row + 1],
            retain_depth_candidates=retain_depth,
            support_mask=mask[row : row + 1],
        )
        for index in (0, 1):
            torch.testing.assert_close(result[index][row : row + 1], single[index], equal_nan=True)
        for key, value in result[2].items():
            torch.testing.assert_close(value[row : row + 1], single[2][key], equal_nan=True)


def test_candidate_assembly_and_refinement_do_not_read_tensor_values_on_host(monkeypatch):
    state, target, sizes, hulls, mask = cases()

    def pockets(q, center, object_q, supports, collision, geometry, **kwargs):
        assert kwargs["support_mask"] is mask
        shape = q.shape[:2]
        return center[:, None] - q[..., :3] * 0.01, {
            "pocket_fit": torch.ones(shape, device="cuda", dtype=torch.bool),
            "antipodal_margin": q.new_full(shape, 0.2),
            "environment_clearance_m": q.new_full(shape, 0.02),
        }

    monkeypatch.setattr(planning, "pocket_pose_targets", pockets)

    def forbidden(*args, **kwargs):
        raise AssertionError("Candidate assembly must remain on the GPU")

    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        result = planning.grasp_candidate_bank(
            state,
            target,
            sizes,
            None,
            None,
            hulls,
            support_mask=mask,
            refine_orientations=True,
            fallback_tool_tilt_deg=30,
        )
    assert result[0].shape[0] == 2
    assert result[0].shape[1] > result[3]
    assert result[4] > 0


def test_prepared_bank_is_consumed_without_regeneration_and_preserves_planned_paths(monkeypatch):
    state, target, sizes, hulls, mask = cases()
    state = {key: value[:1] for key, value in state.items()}
    state["ee_pos"] = target[:1] + target.new_tensor([[-0.2, 0, 0.1]])
    state["dof_pos"] = target.new_zeros(1, 6)
    target, sizes, hulls = target[:1], sizes[:1], hulls[:1]
    bank = (
        state["ee_orn"][:, None],
        target[:, None],
        {
            "pocket_fit": torch.ones(1, 1, device="cuda", dtype=torch.bool),
            "antipodal_margin": target.new_ones(1, 1),
            "environment_clearance_m": target.new_ones(1, 1),
        },
        1,
        0,
    )

    def inverse(p, q, **kwargs):
        return SimpleNamespace(
            exact=torch.ones(len(p), device=p.device, dtype=torch.bool),
            arm_joint_positions=p.new_zeros(len(p), 6),
            position_error=torch.zeros_like(p),
            orientation_error=torch.zeros_like(p),
        )

    monkeypatch.setattr(planning, "solve_pose_targets", inverse)
    monkeypatch.setattr(planning, "grasp_candidate_bank", lambda *a, **kw: bank)
    collision = SimpleNamespace(
        arm_joint_ids=torch.arange(6, device="cuda"),
        clearance=lambda geometry, roots, joints, **kw: roots.new_ones(roots.shape[:-1]),
        gripper_clearance=lambda geometry, p, q: p.new_ones(p.shape[:-1]),
    )
    limits = target.new_tensor([[-3.0, 3.0]] * 6)
    arguments = (state, target, sizes, limits, collision, None, 0.02)
    normal, normal_report = planning.plan_grasp(*arguments, object_support_points=hulls)

    def forbidden(*args, **kwargs):
        raise AssertionError("A prepared bank must not be regenerated")

    monkeypatch.setattr(planning, "grasp_candidate_bank", forbidden)
    cached, cached_report = planning.plan_grasp(
        *arguments, object_support_points=hulls, prepared_candidates=bank
    )
    assert normal is not None and cached is not None
    for name, path in normal.items():
        for expected, actual in zip(path, cached[name]):
            torch.testing.assert_close(actual, expected)
    assert normal_report["selected"] == cached_report["selected"]
    assert normal_report["feasible"] == cached_report["feasible"]
