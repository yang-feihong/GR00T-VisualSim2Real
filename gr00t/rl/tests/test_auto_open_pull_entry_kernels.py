"""Independent Warp/Torch routing parity and invariance tests."""
import pytest
import torch

pytest.importorskip('warp')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')

from gr00t.rl.auto_open import pull_entry_gpu as reference
from gr00t.rl.auto_open import pull_entry_kernels as kernels


def tensor_free(points, obstacles):
    host = reference._Obstacles(obstacles.axes.cpu(), obstacles.support.cpu(), obstacles.valid.cpu())
    return reference._free(points.cpu(), host).to(points.device)


def tensor_visible(start, end, obstacles, *, epsilon=1e-8):
    host = reference._Obstacles(obstacles.axes.cpu(), obstacles.support.cpu(), obstacles.valid.cpu())
    return reference._visible(start.cpu(), end.cpu(), host, epsilon=epsilon).to(start.device)


def bank(batch=3, obstacles=7, planes=37):
    generator = torch.Generator(device='cuda').manual_seed(456)
    axes = torch.randn(batch,obstacles,planes,2,device='cuda',dtype=torch.float64,generator=generator)
    axes = torch.nn.functional.normalize(axes,dim=-1)
    support = torch.rand(batch,obstacles,planes,device='cuda',dtype=torch.float64,generator=generator)*.7+.1
    valid = torch.rand(batch,obstacles,planes,device='cuda',generator=generator) > .15
    points = torch.randn(batch,53,2,device='cuda',dtype=torch.float64,generator=generator)
    end = torch.randn(batch,53,2,device='cuda',dtype=torch.float64,generator=generator)
    return points,end,reference._Obstacles(axes,support,valid)


def test_randomized_bitwise_torch_parity():
    points,end,obstacles=bank()
    assert torch.equal(kernels.free(points,obstacles),tensor_free(points,obstacles))
    for epsilon in (1e-8, 1e-9, 0.):
        assert torch.equal(kernels.visible(points,end,obstacles,epsilon=epsilon),
                           tensor_visible(points,end,obstacles,epsilon=epsilon))


def test_exact_parallel_contact_and_strict_thresholds():
    axes=torch.tensor([[[[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]]],device='cuda',dtype=torch.float64)
    support=torch.ones(1,1,4,device='cuda',dtype=torch.float64)
    valid=torch.ones_like(support,dtype=torch.bool)
    obstacles=reference._Obstacles(axes,support,valid)
    t=torch.tensor(1.-1e-8,device='cuda',dtype=torch.float64)
    before=torch.nextafter(t,torch.full_like(t,-torch.inf))
    after=torch.nextafter(t,torch.full_like(t,torch.inf))
    x=torch.stack((before,t,after,t.new_tensor(1.),t.new_tensor(-1.),t.new_tensor(0.)))
    points=torch.stack((x,torch.zeros_like(x)),-1)[None]
    assert torch.equal(kernels.free(points,obstacles),tensor_free(points,obstacles))
    for delta in (0.,1e-12,-1e-12,1.000001e-12,-1.000001e-12,1.,-1.):
        end=points+points.new_tensor([delta,.3])
        assert torch.equal(kernels.visible(points,end,obstacles),tensor_visible(points,end,obstacles))


def test_bitwise_row_reorder_subset_and_query_chunk_invariance():
    points,end,obstacles=bank(batch=5)
    expected_free=kernels.free(points,obstacles)
    expected_visible=kernels.visible(points,end,obstacles)
    order=torch.tensor([4,1,3,0,2],device='cuda')
    assert torch.equal(kernels.free(points[order],obstacles.select(order)),expected_free[order])
    assert torch.equal(kernels.visible(points[order],end[order],obstacles.select(order)),expected_visible[order])
    for row in range(5):
        ids=torch.tensor([row],device='cuda')
        assert torch.equal(kernels.free(points[ids],obstacles.select(ids)),expected_free[ids])
        assert torch.equal(kernels.visible(points[ids],end[ids],obstacles.select(ids)),expected_visible[ids])
    for size in (1,7,31):
        free=torch.cat([kernels.free(points[:,i:i+size],obstacles) for i in range(0,53,size)],1)
        visible=torch.cat([kernels.visible(points[:,i:i+size],end[:,i:i+size],obstacles) for i in range(0,53,size)],1)
        assert torch.equal(free,expected_free)
        assert torch.equal(visible,expected_visible)


def test_padding_degenerate_and_nonfinite_parity():
    points,end,obstacles=bank(batch=1)
    obstacles.valid[:,0]=False
    obstacles.axes[:,1]=0.
    points[:,0,0]=torch.nan
    points[:,1,0]=torch.inf
    end[:,2,1]=-torch.inf
    assert torch.equal(kernels.free(points,obstacles),tensor_free(points,obstacles))
    assert torch.equal(kernels.visible(points,end,obstacles),tensor_visible(points,end,obstacles))


def test_current_cuda_stream_is_respected():
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        points,end,obstacles=bank(batch=1)
        actual=kernels.visible(points,end,obstacles)
        expected=tensor_visible(points,end,obstacles)
    stream.synchronize()
    assert torch.equal(actual,expected)
