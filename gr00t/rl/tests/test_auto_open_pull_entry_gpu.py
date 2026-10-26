"""Pure geometry tests: no simulator imports or simulation runs."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gr00t.rl.auto_open.pull_entry_gpu import PullEntryGPU, quat_rotate, _route


def fixture(side=1.0, device="cpu", environments=1, max_samples=1024):
    fixed = np.array(
        [[x, y, z] for x in (0.02, 0.10) for y in (-0.025, 0.025) for z in (0.015, 0.025)],
        dtype=np.float32,
    )
    moving = fixed.copy()
    moving[:, 2] -= 0.04

    def geometry(points):
        return SimpleNamespace(points_body=points, convex_parts_body=[points])

    controller = SimpleNamespace(
        fixed_finger_body_idx=0,
        moving_finger_body_idx=1,
        center_geometry={"fixed_finger": geometry(fixed), "moving_finger": geometry(moving)},
        hinge_axis_sign=-side,
        hinge_idx=0,
        door_normal=np.array([1.0, 0.0, 0.0]),
        panel_hinge_to_free_direction=np.array([0.0, side, 0.0]),
        hinge_world=np.zeros(3),
        panel_hinge_to_free_edge=0.5,
        panel_center_world=np.array([0.0, 0.25 * side, 0.8]),
        panel_half_thickness=0.02,
    )
    cache = PullEntryGPU.from_controllers(
        [controller] * environments, device=device, max_samples=max_samples
    )
    ee = torch.tensor([0.7, 0.8 * side, 0.8], device=device).expand(environments, -1).clone()
    q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(environments, -1).clone()
    root = torch.zeros(environments, 13, device=device)
    root[:, 3] = math.cos(-side * math.pi / 4)
    root[:, 6] = math.sin(-side * math.pi / 4)
    body = torch.zeros(environments, 2, 13, device=device)
    body[:, :, :3] = ee[:, None]
    body[:, :, 3] = 1.0
    state = dict(
        ee_pos=ee,
        ee_orn=q,
        root_states=root,
        robot_body_state_w=body,
        door_joint_position=torch.full((environments, 1), 0.3, device=device),
    )
    return cache, state, controller


def plan(cache, state, count=1, **kwargs):
    device = state["ee_pos"].device
    n = len(state["ee_pos"])
    samples = dict(
        height_m=torch.full((n, count), 0.8, device=device),
        insertion_m=torch.full((n, count), 0.04, device=device),
        yaw_rad=torch.zeros(n, count, device=device),
    )
    return cache.plan(
        state,
        torch.arange(n, device=device),
        samples,
        frame_edge_m=torch.full((n,), 0.5, device=device),
        wall_front_m=torch.full((n,), 0.01, device=device),
        **kwargs,
    )


@pytest.mark.parametrize("side", [-1.0, 1.0])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_entry_geometry(side, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cache, state, _ = fixture(side, device)
    result = plan(cache, state)
    assert result["feasible"].all(), {
        k: v.tolist() for k, v in result.items() if k.endswith("safe")
    }
    length = int(result["lengths"][0, 0])
    p, q = result["positions"][0, 0, :length], result["orientations"][0, 0, :length]
    torch.testing.assert_close(p[0], state["ee_pos"][0])
    torch.testing.assert_close(q[0], state["ee_orn"][0])
    torch.testing.assert_close(p[-1], result["endpoint"][0, 0])
    assert torch.linalg.vector_norm(p[1:] - p[:-1], dim=-1).max() <= 0.002001
    face = result["endpoint"] + quat_rotate(result["orientation"], result["face_offset_local"])
    torch.testing.assert_close(
        ((result["contact_point"] - face) * result["normal"]).sum(-1),
        torch.full((1, 1), 0.002, device=device),
        atol=1e-6,
        rtol=1e-5,
    )
    shaft = quat_rotate(result["orientation"], torch.tensor([1.0, 0.0, 0.0], device=device))
    torch.testing.assert_close(
        shaft, torch.tensor([[[0.0, -side, 0.0]]], device=device), atol=1e-6, rtol=1e-5
    )
    # Values checked against the deleted NumPy/SciPy implementation before its
    # removal. The continuous route may differ from its 5 mm raster route.
    torch.testing.assert_close(
        result["endpoint"][0, 0],
        torch.tensor([0.09428852, 0.56007230 * side, 0.8], device=device),
        atol=2e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        result["ee_tangent_radius_m"],
        torch.full((1, 1), 0.5629217, device=device),
        atol=2e-6,
        rtol=1e-6,
    )


def test_all_eight_candidates_and_hardware_rejection():
    cache, state, _ = fixture(environments=2)
    result = plan(cache, state, 8, hardware_height_bounds=torch.tensor([[0.75, 0.85], [1.0, 1.1]]))
    assert result["positions"].shape == (2, 8, 1024, 3)
    assert not result["feasible"][0].any()
    assert result["feasible"][1].all()


def test_overflow_fails_closed():
    cache, state, _ = fixture(max_samples=8)
    result = plan(cache, state)
    assert result["overflow"].all() and not result["feasible"].any()


def test_visibility_graph_routes_around_rectangle():
    start = torch.tensor([[-1.0, 0.0]])
    end = torch.tensor([[1.0, 0.0]])
    offset = torch.tensor(
        [[[-0.05, -0.05, 0.0], [-0.05, 0.05, 0.0], [0.05, -0.05, 0.0], [0.05, 0.05, 0.0]]]
    )
    rectangle = torch.tensor([[[-0.1, -0.2], [-0.1, 0.2], [0.1, -0.2], [0.1, 0.2]]])
    route, length, safe = _route(start, end, [offset], [rectangle])
    assert safe.all() and length[0] >= 3
    torch.testing.assert_close(route[0, 0], start[0])
    torch.testing.assert_close(route[0, length[0] - 1], end[0])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batch_order_subset_and_chunk_invariance(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cache, state, _ = fixture(device=device, environments=2)
    state["ee_pos"][1, 1] += 0.01
    state["robot_body_state_w"][1, :, 1] += 0.01
    samples = dict(
        height_m=torch.tensor([[0.78, 0.81], [0.80, 0.82]], device=device),
        insertion_m=torch.tensor([[0.02, 0.06], [0.03, 0.05]], device=device),
        yaw_rad=torch.tensor([[-0.02, 0.02], [0.01, -0.01]], device=device),
    )

    def run(rows, chunk):
        cache.candidate_chunk = chunk
        rows = torch.tensor(rows, device=device)
        return cache.plan(
            state,
            rows,
            {k: v[rows] for k, v in samples.items()},
            frame_edge_m=torch.full((len(rows),), 0.5, device=device),
            wall_front_m=torch.full((len(rows),), 0.01, device=device),
        )

    baseline = run([0, 1], 128)
    reordered = run([1, 0], 1)
    single = run([1], 2)
    assert torch.equal(baseline["positions"][:, :, 0], state["ee_pos"][:, None].expand(-1, 2, -1))
    assert torch.equal(
        baseline["orientations"][:, :, 0], state["ee_orn"][:, None].expand(-1, 2, -1)
    )
    for key, value in baseline.items():
        if value.dtype in (torch.bool, torch.long):
            assert torch.equal(value, reordered[key].flip(0)), key
            assert torch.equal(value[1:2], single[key]), key
        else:
            assert torch.equal(value, reordered[key].flip(0)), key
            assert torch.equal(value[1:2], single[key]), key


def test_empty_rows_keep_fixed_axes():
    cache, state, _ = fixture()
    empty = torch.empty(0, 8)
    result = cache.plan(
        state,
        torch.empty(0, dtype=torch.long),
        dict(height_m=empty, insertion_m=empty, yaw_rad=empty),
        frame_edge_m=torch.empty(0),
        wall_front_m=torch.empty(0),
    )
    assert result["positions"].shape == (0, 8, 1024, 3)
    assert result["feasible"].shape == (0, 8)


def test_separate_convex_parts_do_not_fill_empty_jaw_space():
    from gr00t.rl.auto_open.pull_entry_gpu import _halfspaces, _free

    lower = torch.tensor([[[-0.02, -0.12], [-0.02, -0.08], [0.02, -0.12], [0.02, -0.08]]])
    upper = lower + torch.tensor([0.0, 0.20])
    rectangle = torch.tensor([[[-0.01, -0.01], [-0.01, 0.01], [0.01, -0.01], [0.01, 0.01]]])
    points = torch.tensor([[[0.0, 0.0], [0.0, 0.10]]])
    assert _free(
        points, [_halfspaces(lower, rectangle), _halfspaces(upper, rectangle)]
    ).tolist() == [[True, False]]
    assert _free(points, [_halfspaces(torch.cat((lower, upper), 1), rectangle)]).tolist() == [
        [False, False]
    ]
