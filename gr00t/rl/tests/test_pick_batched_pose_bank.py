"""Different object sizes generate their own GPU pinch families in a common batch."""

import pytest
import torch

from gr00t.rl.auto_open.motion_math import _euler_xyz_quat
from gr00t.rl.pick.planning import grasp_pose_bank, object_frame_grasp_bank

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU candidate generation")


@pytest.mark.parametrize("transport", [False, True])
def test_heterogeneous_pose_candidates_match_each_independent_object(monkeypatch, transport):
    target = torch.tensor([[0.6, 0.1, 0.03], [2.5, 1.0, 0.8]], device="cuda")
    origins = torch.tensor([[0.0, 0.0, 0.7], [2.0, 1.0, 0.7]], device="cuda")
    sizes = torch.tensor([[0.04, 0.06, 0.05], [0.07, 0.05, 0.12]], device="cuda")
    heading = torch.tensor([0.1, 0.4], device="cuda")
    reference = _euler_xyz_quat(target.new_tensor([[0.0, 0.5, 0.2], [0.1, 0.2, 0.4]]))
    options = {"reference_orientation": reference} if transport else {}
    with monkeypatch.context() as scope:

        def no_read(*args, **kwargs):
            raise AssertionError("Candidate generation must stay on the device")

        scope.setattr(torch.Tensor, "item", no_read)
        scope.setattr(torch.Tensor, "cpu", no_read)
        batched = grasp_pose_bank(target, heading, sizes, origins, **options)
    for row in range(2):
        single = grasp_pose_bank(
            target[row : row + 1],
            heading[row : row + 1],
            sizes[row],
            origins[row : row + 1],
            **({"reference_orientation": reference[row : row + 1]} if transport else {}),
        )
        for actual, expected in zip(batched, single):
            torch.testing.assert_close(actual[row : row + 1], expected)


def test_object_frame_banks_do_not_reuse_the_first_objects_aperture():
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device="cuda")
    sizes = torch.tensor([[0.04, 0.04, 0.05], [0.07, 0.07, 0.10]], device="cuda")
    bank = object_frame_grasp_bank(q, sizes)
    assert not torch.allclose(bank[0], bank[1])
    for row in range(2):
        torch.testing.assert_close(
            bank[row : row + 1], object_frame_grasp_bank(q[row : row + 1], sizes[row])
        )
