"""Basket segment timing preserves each row's speed limits and valid path frames."""

import pytest
import torch

from gr00t.rl.auto_open.motion_math import _euler_xyz_quat
from gr00t.rl.pick.basket_nudge import pose_segments, pose_segments_batch


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("contact_segment", [None, 1])
def test_segment_batch_matches_split_and_permutation_without_host_reads(
    device, contact_segment, monkeypatch
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU segment timing")
    points = torch.tensor(
        [
            [[0, 0, 0], [0.1, 0, 0.2], [0.15, 0, 0.2]],
            [[10, 0, 0.5], [10.03, 0.1, 0.6], [10.04, 0.1, 0.6]],
        ],
        device=device,
        dtype=torch.float32,
    )
    q = _euler_xyz_quat(
        points.new_tensor(
            [
                [[0.1, 0.2, 0.3], [0.2, 0.4, 0.5], [0.4, 0.3, 0.6]],
                [[-0.2, 0.1, 1.2], [-0.1, 0.3, 1.1], [-0.1, 0.3, 1.1]],
            ]
        )
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Segment timing must not read tensor scalars on the host")

    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        path, valid, boundaries = pose_segments_batch(
            points, q, 0.02, contact_segment=contact_segment
        )
    assert valid.all() and path[2][0] != path[2][1]
    for row in range(2):
        single_p, single_q, bounds = pose_segments(
            points[row : row + 1], q[row : row + 1], 0.02, contact_segment=contact_segment
        )
        length = int(path[2][row])
        assert length == single_p.shape[1]
        assert boundaries[row].tolist() == bounds
        torch.testing.assert_close(path[0][row, :length], single_p[0], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(path[1][row, :length], single_q[0], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(path[0][row, length:], points[row, -1].expand(4096 - length, -1))
        for segment in range(2):
            start, stop = bounds[segment : segment + 2]
            speed = 0.025 if segment == contact_segment else 0.15
            assert path[0][row, start:stop].diff(dim=0).norm(dim=-1).max() / 0.02 <= speed + 1e-5
    reverse = torch.tensor([1, 0], device=device)
    reordered = pose_segments_batch(
        points[reverse], q[reverse], 0.02, contact_segment=contact_segment
    )
    for expected, actual in zip(path, reordered[0]):
        torch.testing.assert_close(expected, actual[reverse])
    limited, limited_valid, _ = pose_segments_batch(points, q, 0.02, capacity=2)
    assert not limited_valid.any() and limited[2].tolist() == [2, 2]


def test_invalid_segment_shape_is_rejected_before_path_building():
    with pytest.raises(ValueError, match="Segment paths require"):
        pose_segments_batch(torch.zeros(2, 2, 4), torch.zeros(2, 2, 4), 0.02)
