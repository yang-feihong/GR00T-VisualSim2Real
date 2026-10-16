import pytest
import torch

from gr00t.rl.third_party.visual_wholebody_lowlevel.legged_gym.utils.batch_invariant_math import (
    fixed_dot,
    fixed_mean,
    fixed_norm,
    fixed_normalize,
    fixed_sum,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("width", [2, 3, 4, 6, 7])
def test_fixed_order_math_is_exactly_independent_of_batch_shape(device, width):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    generator = torch.Generator(device=device).manual_seed(1702)
    row = torch.randn((1, width), generator=generator, device=device)
    other = torch.randn((8, width), generator=generator, device=device)
    batch = torch.cat((row, other), dim=0)

    operations = (
        lambda value: fixed_sum(value),
        lambda value: fixed_mean(value),
        lambda value: fixed_dot(value, value),
        lambda value: fixed_norm(value),
        lambda value: fixed_normalize(value),
    )
    for operation in operations:
        assert torch.equal(operation(row)[0], operation(batch)[0])


def test_fixed_order_math_supports_nonfinal_dimensions_and_keepdim():
    value = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    torch.testing.assert_close(fixed_sum(value, dim=1), value.sum(dim=1))
    assert fixed_mean(value, dim=1, keepdim=True).shape == (2, 1, 4)
    assert fixed_norm(value, dim=1, keepdim=True).shape == (2, 1, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("width", [2, 3, 4, 6, 7])
def test_fixed_order_math_preserves_native_single_row_results(width):
    generator = torch.Generator(device="cuda").manual_seed(1800 + width)
    left = torch.randn((256, width), generator=generator, device="cuda")
    right = torch.randn((256, width), generator=generator, device="cuda")
    for row in range(len(left)):
        one_left = left[row : row + 1]
        one_right = right[row : row + 1]
        assert torch.equal(
            fixed_norm(one_left), torch.linalg.vector_norm(one_left, dim=-1)
        )
        assert torch.equal(
            fixed_dot(one_left, one_right),
            torch.sum(one_left * one_right, dim=-1),
        )
