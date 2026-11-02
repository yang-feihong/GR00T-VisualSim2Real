"""Sparse screening must not materialize every full-length candidate."""

import pytest
import torch

from gr00t.rl.pick import gaze
from gr00t.rl.tests.test_pick_batched_gaze import inputs, install_reference_tools


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sparse_anchor_and_selected_candidate_generation_match_dense(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    roots, initial, target, limits = inputs(device)
    roots = roots[:, :1].expand(-1, 83, -1).clone()
    offsets = roots.new_tensor([[0.03, -0.02, 0.1]]).expand(3, -1)
    anchors = torch.tensor([0, 7, 25, 75, 82], device=device)
    times = (anchors * 0.02)[None].expand(3, -1)
    choice = torch.tensor([[0, 104, 333], [7, 115, 512], [22, 600, 615]], device=device)
    dense = gaze.natural_gaze_candidates(
        roots, initial, target, limits, 0.02, offsets, joint_sampling=True
    )
    sparse = gaze.natural_gaze_candidates(
        roots[:, anchors],
        initial,
        target,
        limits,
        0.02,
        offsets,
        sample_times=times,
        joint_sampling=True,
    )
    selected = gaze.natural_gaze_candidates(
        roots,
        initial,
        target,
        limits,
        0.02,
        offsets,
        candidate_indices=choice,
        joint_sampling=True,
    )
    candidates = len(dense[4])
    assert candidates == 616 and candidates <= 1000
    for full, short, chosen in zip(dense[:4], sparse[:4], selected[:4], strict=True):
        full = full.reshape(3, candidates, 83, -1)
        torch.testing.assert_close(
            short.reshape(3, candidates, len(anchors), -1), full[:, :, anchors]
        )
        expected = full.gather(1, choice[..., None, None].expand(-1, -1, 83, full.shape[-1]))
        torch.testing.assert_close(chosen.reshape(3, 3, 83, -1), expected)


def test_sparse_planner_builds_only_four_full_paths(monkeypatch):
    roots, initial, target, limits = inputs("cpu")
    roots = roots[:, :1].expand(-1, 83, -1).clone()
    original = gaze.natural_gaze_candidates
    original_offsets = gaze.gaze_candidate_offsets
    collision, _ = install_reference_tools(monkeypatch)
    monkeypatch.setattr(gaze, "gaze_candidate_offsets", original_offsets)
    requests = []

    def trace(body, *args, **kwargs):
        indices = kwargs.get("candidate_indices")
        requests.append((body.shape[1], None if indices is None else indices.shape[1]))
        return original(body, *args, **kwargs)

    monkeypatch.setattr(gaze, "natural_gaze_candidates", trace)
    gaze.plan_approach_gaze_batch(
        roots, initial, target, limits, 0.02, collision, roots[:, 0, 0], roots.new_zeros(3, 3)
    )
    assert requests == [(9, None), (83, 4)]
