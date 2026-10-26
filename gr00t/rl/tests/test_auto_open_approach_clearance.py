import math

import pytest
import torch

from gr00t.rl.auto_open.geometry import initial_approach_clearance_gpu


def fixture(device):
    front = torch.tensor([0.80, 0.83, 0.90, 0.90, 0.90, 0.90], dtype=torch.float64, device=device)
    gap = front.new_tensor([0.0, 0.01, 0.01, 0.02, 0.0201, 0.10])
    root = front.new_zeros(6, 13)
    root[:, 3] = 1.0
    bodies = front.new_zeros(6, 2, 13)
    bodies[:, :, 3] = 1.0
    bodies[:, 0, 0] = front
    bodies[:, 1, 0] = front - 0.01
    handle = front.new_zeros(6, 1, 13)
    handle[:, :, 3] = 1.0
    handle[:, 0, 2] = gap
    cache = dict(
        fixed_body_idx=0,
        moving_body_idx=1,
        handle_body_idx=0,
        fixed_points=front.new_zeros(1, 3),
        moving_points=front.new_zeros(1, 3),
        handle_center_local=front.new_zeros(6, 3),
        handle_axis_local=front.new_tensor([[0.0, 1.0, 0.0]]).expand(6, -1),
        handle_half_length=front.new_zeros(6),
        handle_radius=front.new_zeros(6),
    )
    return dict(root_states=root, robot_body_state_w=bodies, door_body_state_w=handle), cache


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_approach_only_increases_for_strictly_close_heights(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    state, cache = fixture(device)
    result = initial_approach_clearance_gpu(state, cache)
    torch.testing.assert_close(
        result["approach_distance_m"],
        state["root_states"].new_tensor([0.85, 0.85, 0.92, 0.85, 0.85, 0.85]),
    )
    assert result["height_near"].tolist() == [True, True, True, False, False, False]
    assert result["enlarged"].tolist() == [False, False, True, False, False, False]
    one_state = {name: value[2:3] for name, value in state.items()}
    one_cache = {
        name: value[2:3] if name.startswith("handle_") and torch.is_tensor(value) else value
        for name, value in cache.items()
    }
    single = initial_approach_clearance_gpu(one_state, one_cache)
    for name, value in result.items():
        torch.testing.assert_close(single[name], value[2:3], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("side", [-1.0, 1.0])
def test_forward_extent_is_base_relative_and_handle_radius_affects_gap(side):
    state, cache = fixture("cpu")
    root = state["root_states"]
    root[:, :3] = root.new_tensor([4.0, -3.0, 0.5])
    root[:, 3] = math.sqrt(0.5)
    root[:, 6] = side * math.sqrt(0.5)
    bodies = state["robot_body_state_w"]
    x = bodies[:, :, 0].clone()
    bodies[:, :, :3] = root[:, None, :3]
    bodies[:, :, 1] += side * x
    bodies[:, :, 3:7] = root[:, None, 3:7]
    state["door_body_state_w"][:, 0, 2] += 0.5
    cache["handle_radius"].fill_(0.01)
    result = initial_approach_clearance_gpu(state, cache)
    torch.testing.assert_close(result["gripper_front_m"], x[:, 0])
    torch.testing.assert_close(
        result["height_gap_m"], root.new_tensor([0.0, 0.0, 0.0, 0.01, 0.0101, 0.09])
    )
    torch.testing.assert_close(
        result["approach_distance_m"], root.new_tensor([0.85, 0.85, 0.92, 0.92, 0.92, 0.85])
    )
