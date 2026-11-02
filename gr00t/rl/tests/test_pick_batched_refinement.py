"""Candidate refinement never borrows anchors or scene queries from another row."""

from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.auto_open.motion_math import _euler_xyz_quat
from gr00t.rl.pick import planning

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU grasp planning")


@pytest.mark.parametrize("shared_cloud", [False, True])
def test_direct_pocket_queries_cap_surface_points_on_device(monkeypatch, shared_cloud):
    count, width = 2, 12_000
    points = torch.arange(count * width * 3, device="cuda", dtype=torch.float32)
    points = points.reshape(count, width, 3)
    mask = torch.ones(count, width, device="cuda", dtype=torch.bool)
    if shared_cloud:
        points, mask = points[0], None
    q = points.new_tensor([1, 0, 0, 0]).expand(count, 1, -1)
    targets = points.new_zeros(count, 3)

    def quality(orientations, object_orientation, support, **kwargs):
        assert support.shape == (count, 9984, 3)
        assert support.is_cuda and kwargs["support_mask"].is_cuda
        assert kwargs["support_mask"].all()
        torch.testing.assert_close(
            support[:, 0], points[0].expand(count, 3) if shared_cloud else points[:, 0]
        )
        torch.testing.assert_close(
            support[:, -1], points[-1].expand(count, 3) if shared_cloud else points[:, -1]
        )
        return {
            "center_local": targets[:, None].clone(),
            "environment_clearance_m": targets.new_ones(count, 1),
        }

    monkeypatch.setattr(planning, "convex_grasp_quality", quality)
    goals, _ = planning.pocket_pose_targets(
        q, targets, q[:, 0], points, None, None, support_mask=mask
    )
    assert goals.shape == (count, 1, 3)


def test_refinement_matches_independent_rows_and_masks_unsupported_pairs(monkeypatch):
    q = _euler_xyz_quat(
        torch.tensor(
            [
                [[0, 0, 0], [0, 0, 0.5], [0, 0, 1.0], [0, 0, 1.5]],
                [[0, 0, 0.2], [0, 0, 0.7], [0, 0, 1.2], [0, 0, 1.7]],
            ],
            device="cuda",
            dtype=torch.float32,
        )
    )
    quality = {
        "pocket_fit": torch.tensor(
            [[True, True, True, False], [False, True, False, False]], device="cuda"
        ),
        "antipodal_margin": torch.full((2, 4), 0.2, device="cuda"),
        "environment_clearance_m": torch.full((2, 4), 0.02, device="cuda"),
    }

    def pockets(orientations, *args, **kwargs):
        return orientations[..., :3], {
            "pocket_fit": torch.ones(orientations.shape[:2], device="cuda", dtype=torch.bool),
            "antipodal_margin": torch.full(orientations.shape[:2], 0.3, device="cuda"),
            "environment_clearance_m": torch.full(orientations.shape[:2], 0.01, device="cuda"),
        }

    monkeypatch.setattr(planning, "pocket_pose_targets", pockets)
    goals = torch.zeros(2, 4, 3, device="cuda")
    targets = torch.zeros(2, 3, device="cuda")
    hulls = torch.zeros(2, 8, 3, device="cuda")
    mask = torch.ones(2, 8, device="cuda", dtype=torch.bool)
    batch = planning.refine_grasp_candidates(
        q, goals, quality, q[:, 0], targets, q[:, 0], hulls, None, None, support_mask=mask
    )
    for row in range(2):
        single = planning.refine_grasp_candidates(
            q[row : row + 1],
            goals[row : row + 1],
            {key: value[row : row + 1] for key, value in quality.items()},
            q[row : row + 1, 0],
            targets[row : row + 1],
            q[row : row + 1, 0],
            hulls[row : row + 1],
            None,
            None,
            support_mask=mask[row : row + 1],
        )
        torch.testing.assert_close(batch[0][row : row + 1], single[0])
        torch.testing.assert_close(batch[1][row : row + 1], single[1])
        for key in quality:
            torch.testing.assert_close(batch[2][key][row : row + 1], single[2][key])
    assert batch[2]["pocket_fit"][0, 4:].sum() == 27
    assert not batch[2]["pocket_fit"][1, 4:].any()


def test_pocket_memory_chunks_bound_rows_and_candidates_without_losing_scene_ownership(monkeypatch):
    count, candidates, points = 5, 3, 4
    budget = 2 * points
    monkeypatch.setattr(planning, "MAX_POCKET_SAMPLE_EVALUATIONS", budget)
    calls = []

    class Geometry:
        supports_query_mask = True

        def __init__(self, owners):
            self.owners = owners

        def select(self, rows):
            return Geometry(self.owners[rows])

    def clearance(geometry, p, q, *args, **kwargs):
        return geometry.owners[:, None].float().expand(p.shape[:2]) + 0.01

    def quality(q, object_q, supports, **kwargs):
        batch, width = q.shape[:2]
        calls.append((batch, width))
        # The CUDA fused contact kernel does not materialize the depth axis.
        assert batch * width * supports.shape[-2] <= budget
        center = q.new_zeros(batch, width, 1, 3)
        aperture = q.new_ones(batch, width, 1)
        gap = kwargs["depth_clearance"](
            center, aperture, torch.ones_like(aperture, dtype=torch.bool)
        )
        return {"center_local": center[:, :, 0], "environment_clearance_m": gap[..., 0]}

    monkeypatch.setattr(planning, "convex_grasp_quality", quality)
    q = torch.tensor([1.0, 0, 0, 0], device="cuda").expand(count, candidates, -1)
    targets = torch.zeros(count, 3, device="cuda")
    hulls = torch.zeros(count, points, 3, device="cuda")
    geometry = Geometry(torch.arange(count, device="cuda"))
    collision = SimpleNamespace(gripper_clearance=clearance, gripper_closing_clearance=clearance)
    goals, report = planning.pocket_pose_targets(q, targets, q[:, 0], hulls, collision, geometry)
    assert len(calls) > 1
    assert goals.shape == (count, candidates, 3)
    expected = torch.arange(count, device="cuda")[:, None].expand(-1, candidates) + 0.01
    torch.testing.assert_close(report["environment_clearance_m"], expected)


def test_compact_depth_queries_keep_scene_owners_and_padded_hulls(monkeypatch):
    owners_seen = []

    class Geometry:
        def select(self, owners):
            owners_seen.append(owners.clone())
            return SimpleNamespace(owners=owners)

    def clearance(geometry, positions, quats, *args, maximum_distance_m):
        assert maximum_distance_m == pytest.approx(0.0021)
        assert positions.shape == (3, 1, 3)
        return geometry.owners[:, None].float() + 0.01

    hulls = torch.zeros(2, 8, 3, device="cuda")
    mask = torch.ones(2, 8, device="cuda", dtype=torch.bool)

    def quality(orientations, object_orientation, support_points, **kwargs):
        assert kwargs["support_mask"] is mask
        assert support_points is hulls
        feasible = torch.tensor(
            [[[True, False], [False, False]], [[False, True], [True, False]]], device="cuda"
        )
        result = kwargs["depth_clearance"](
            torch.zeros(2, 2, 2, 3, device="cuda"),
            torch.ones(2, 2, 2, device="cuda"),
            feasible,
        )
        torch.testing.assert_close(result[feasible], result.new_tensor([0.01, 1.01, 1.01]))
        assert result[~feasible].isnan().all()
        return {
            "center_local": torch.zeros(2, 2, 3, device="cuda"),
            "environment_clearance_m": result[..., 0],
        }

    monkeypatch.setattr(planning, "convex_grasp_quality", quality)
    rotations = torch.tensor([1, 0, 0, 0], device="cuda", dtype=torch.float32).expand(2, 2, 4)
    planning.pocket_pose_targets(
        rotations,
        torch.zeros(2, 3, device="cuda"),
        rotations[:, 0],
        hulls,
        SimpleNamespace(gripper_clearance=clearance, gripper_closing_clearance=clearance),
        Geometry(),
        support_mask=mask,
    )
    torch.testing.assert_close(owners_seen[0], torch.tensor([0, 1, 1], device="cuda"))


def test_masked_depth_queries_do_not_compact_candidates(monkeypatch):
    masks = []
    feasible = torch.tensor(
        [[[True, False], [False, False]], [[False, True], [True, False]]], device="cuda"
    )

    def clearance(geometry, positions, rotations, *args, query_mask, maximum_distance_m):
        assert maximum_distance_m == pytest.approx(0.0021)
        assert positions.shape == (2, 4, 3)
        assert rotations.shape == (2, 4, 4)
        masks.append(query_mask)
        return torch.ones(2, 4, device="cuda")

    def quality(orientations, object_orientation, support_points, **kwargs):
        result = kwargs["depth_clearance"](
            torch.zeros(2, 2, 2, 3, device="cuda"),
            torch.ones(2, 2, 2, device="cuda"),
            feasible,
        )
        assert result[~feasible].isnan().all()
        assert (result[feasible] == 1).all()
        return {
            "center_local": torch.zeros(2, 2, 3, device="cuda"),
            "environment_clearance_m": result[..., 0],
        }

    monkeypatch.setattr(planning, "convex_grasp_quality", quality)
    rotations = torch.tensor([1, 0, 0, 0], device="cuda", dtype=torch.float32).expand(2, 2, 4)
    planning.pocket_pose_targets(
        rotations,
        torch.zeros(2, 3, device="cuda"),
        rotations[:, 0],
        torch.zeros(8, 3, device="cuda"),
        SimpleNamespace(gripper_clearance=clearance, gripper_closing_clearance=clearance),
        SimpleNamespace(supports_query_mask=True),
    )
    assert len(masks) == 2
    for mask in masks:
        torch.testing.assert_close(mask, feasible.flatten(1, 2))
