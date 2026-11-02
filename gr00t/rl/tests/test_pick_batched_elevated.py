"""Elevated measured-grasp transitions preserve roots, seeds and scene ownership."""

from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.pick import planning
from gr00t.rl.pick.planning import elevated_cartesian_approach
from gr00t.rl.tests.test_pick_batched_grasp_inverse import case, fake_inverse

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU elevated approach")


@pytest.mark.parametrize("native", [False, True])
def test_elevated_approach_matches_split_with_each_rows_joint_seed_and_scene(native, monkeypatch):
    state, bank, limits = case()
    if not native:
        monkeypatch.setattr(planning, "solve_pose_targets", fake_inverse)
    p, q = bank[1][:, 0], bank[0][:, 0]
    heights = state["root_states"][:, 2] + 0.01

    def arm_gap(geometry, roots, joints, **kwargs):
        if geometry is not None:
            expected = state["root_states"][geometry.owners, 0]
            torch._assert_async(
                (roots[..., 0] - expected[:, None, None, None]).abs().max() < 1,
                "Elevated root queries crossed scene owners",
            )
        return torch.where(roots[..., 2] < 0.25, -0.01, 0.02)

    collision = SimpleNamespace(
        arm_joint_ids=torch.arange(6, device="cuda"),
        clearance=arm_gap,
        gripper_clearance=lambda geometry, positions, quats: positions[..., 2] + 0.05,
    )
    geometry = SimpleNamespace(owners=torch.arange(3, device="cuda"))
    geometry.select = lambda rows: SimpleNamespace(owners=geometry.owners[rows])

    def forbidden(*args, **kwargs):
        raise AssertionError("Elevated transitions must remain on the GPU")

    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        result = elevated_cartesian_approach(state, p, q, heights, limits, collision, geometry, 151)
    for row in range(3):
        split = elevated_cartesian_approach(
            {key: value[row : row + 1] for key, value in state.items()},
            p[row : row + 1],
            q[row : row + 1],
            heights[row : row + 1],
            limits,
            collision,
            None,
            151,
        )
        for together, separate in zip(result, split):
            torch.testing.assert_close(together[row : row + 1], separate)
