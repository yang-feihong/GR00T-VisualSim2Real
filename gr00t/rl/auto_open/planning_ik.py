"""Replay the existing candidate IK math without per-operation host dispatch."""

from collections import OrderedDict
import threading

import torch

from gr00t.rl.envs.door.b2z1_reset_geometry import solve_b2z1_arm_ik as _solve_ik


_CACHE = OrderedDict()
_LOCK = threading.Lock()
_MAX_GRAPHS = 8


class _CandidateIKGraph:
    def __init__(self, inputs, iterations, damping):
        self.inputs = tuple(value.clone() for value in inputs)
        self.status = torch.zeros((), device=inputs[0].device, dtype=torch.bool)
        self.done = torch.cuda.Event()
        current = torch.cuda.current_stream(inputs[0].device)
        stream = torch.cuda.Stream(device=inputs[0].device)
        stream.wait_stream(current)

        def compute():
            self.status.zero_()
            return _solve_ik(
                *self.inputs, iterations=iterations, damping=damping,
                error_status=self.status,
            )

        with torch.cuda.stream(stream):
            compute()
        current.wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.outputs = compute()
        current.wait_stream(stream)
        self.done.record(current)

    def replay(self, inputs):
        stream = torch.cuda.current_stream(inputs[0].device)
        stream.wait_event(self.done)
        for target, value in zip(self.inputs, inputs, strict=True):
            target.copy_(value)
        self.graph.replay()
        outputs = tuple(value.clone() for value in self.outputs)
        self.done.record(stream)
        if bool(self.status.item()):
            raise RuntimeError("candidate IK linear solve failed during CUDA graph replay")
        return outputs


def solve_candidate_ik(*inputs, iterations=80, damping=0.03):
    """Keep sampling, iteration count, clamping and acceptance exactly unchanged."""
    if not inputs[0].is_cuda or not inputs[0].numel():
        return _solve_ik(*inputs, iterations=iterations, damping=damping)
    key = (
        tuple((value.device, value.dtype, tuple(value.shape), value.stride()) for value in inputs),
        int(iterations), float(damping),
    )
    with _LOCK:
        entry = _CACHE.get(key)
        if entry is None:
            entry = _CandidateIKGraph(inputs, iterations, damping)
            _CACHE[key] = entry
            if len(_CACHE) > _MAX_GRAPHS:
                _, evicted = _CACHE.popitem(last=False)
                evicted.done.synchronize()
        _CACHE.move_to_end(key)
        return entry.replay(inputs)
