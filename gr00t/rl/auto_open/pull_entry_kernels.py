"""Fused GPU pull-entry halfspace kernels.

These accept the frozen planner's packed float64 routing obstacles. Each CUDA
thread owns one (row, query, obstacle); only a boolean per obstacle is stored,
not a [row, query, obstacle, plane] intermediate. No block/global floating-point
reduction or cuBLAS operation is used. FMA and fast math are disabled to retain
the explicit Torch operation order at the existing strict-interior thresholds.

The online planner uses these kernels for all CUDA routing queries.
"""

import torch
import warp as wp

wp.set_module_options({"enable_backward": False, "enable_fma": False, "fast_math": False})


@wp.kernel
def _free_f64(
    points: wp.array(dtype=wp.float64, ndim=3),
    axes: wp.array(dtype=wp.float64, ndim=4),
    support: wp.array(dtype=wp.float64, ndim=3),
    valid: wp.array(dtype=wp.bool, ndim=3),
    result: wp.array(dtype=wp.bool, ndim=3),
):
    row, query, obstacle = wp.tid()
    outside = wp.bool(False)
    for plane in range(support.shape[2]):
        if valid[row, obstacle, plane]:
            x = points[row, query, 0] * axes[row, obstacle, plane, 0]
            y = points[row, query, 1] * axes[row, obstacle, plane, 1]
            d = (x + y) - support[row, obstacle, plane]
            # Torch's ~(d < threshold), not d >= threshold: retain NaN behavior.
            if not (d < wp.float64(-1e-8)):
                outside = True
                break
    result[row, query, obstacle] = outside


@wp.kernel
def _visible_f64(
    start: wp.array(dtype=wp.float64, ndim=3),
    end: wp.array(dtype=wp.float64, ndim=3),
    axes: wp.array(dtype=wp.float64, ndim=4),
    support: wp.array(dtype=wp.float64, ndim=3),
    valid: wp.array(dtype=wp.bool, ndim=3),
    epsilon: wp.float64,
    result: wp.array(dtype=wp.bool, ndim=3),
):
    row, query, obstacle = wp.tid()
    lower = wp.float64(0.0)
    upper = wp.float64(1.0)
    outside = wp.bool(False)
    nan_interval = wp.bool(False)
    dx = end[row, query, 0] - start[row, query, 0]
    dy = end[row, query, 1] - start[row, query, 1]
    for plane in range(support.shape[2]):
        if valid[row, obstacle, plane]:
            nx, ny = axes[row, obstacle, plane, 0], axes[row, obstacle, plane, 1]
            x, y = start[row, query, 0] * nx, start[row, query, 1] * ny
            d = (x + y) - support[row, obstacle, plane]
            sx, sy = dx * nx, dy * ny
            slope = sx + sy
            if wp.abs(slope) < wp.float64(1e-12):
                if d >= -epsilon:
                    outside = True
                    break
            elif slope < wp.float64(-1e-12):
                crossing = (-epsilon - d) / slope
                nan_interval = nan_interval or wp.isnan(crossing)
                if crossing > lower:
                    lower = crossing
            elif slope > wp.float64(1e-12):
                crossing = (-epsilon - d) / slope
                nan_interval = nan_interval or wp.isnan(crossing)
                if crossing < upper:
                    upper = crossing
    result[row, query, obstacle] = outside or (
        (not nan_interval) and lower >= upper - wp.float64(1e-8)
    )


def _arguments(points, obstacles):
    wp.init()
    if points.device.type != "cuda" or points.dtype != torch.float64:
        raise ValueError("Fused routing kernels require CUDA float64 query tensors")
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError("Expected query shape [batch, queries, 2]")
    if obstacles.axes.dtype != torch.float64 or obstacles.support.dtype != torch.float64:
        raise ValueError("Packed routing planes must be float64")
    if obstacles.valid.dtype != torch.bool:
        raise ValueError("Packed plane validity must be bool")
    for value in (obstacles.axes, obstacles.support, obstacles.valid):
        if value.device != points.device or value.shape[0] != points.shape[0]:
            raise ValueError("Queries and packed planes must share batch and CUDA device")
    return (
        wp.from_torch(obstacles.axes, dtype=wp.float64),
        wp.from_torch(obstacles.support, dtype=wp.float64),
        wp.from_torch(obstacles.valid, dtype=wp.bool),
    )


def free(points, obstacles):
    """Drop-in for _free with already packed obstacles; returns bool [B,Q]."""
    axes, support, valid = _arguments(points, obstacles)
    result = torch.empty(
        (*points.shape[:2], obstacles.support.shape[1]), dtype=torch.bool, device=points.device
    )
    if result.numel():
        wp.launch(
            _free_f64,
            dim=result.shape,
            inputs=[wp.from_torch(points, dtype=wp.float64), axes, support, valid],
            outputs=[wp.from_torch(result, dtype=wp.bool)],
            stream=wp.stream_from_torch(torch.cuda.current_stream(points.device)),
        )
    return result.all(-1)


def visible(start, end, obstacles, *, epsilon=1e-8):
    """Drop-in for _visible with packed float64 obstacles; bool [B,Q]."""
    axes, support, valid = _arguments(start, obstacles)
    if end.shape != start.shape or end.dtype != start.dtype or end.device != start.device:
        raise ValueError("Segment endpoints must have identical shape, dtype and device")
    result = torch.empty(
        (*start.shape[:2], obstacles.support.shape[1]), dtype=torch.bool, device=start.device
    )
    if result.numel():
        wp.launch(
            _visible_f64,
            dim=result.shape,
            inputs=[
                wp.from_torch(start, dtype=wp.float64),
                wp.from_torch(end, dtype=wp.float64),
                axes,
                support,
                valid,
                wp.float64(epsilon),
            ],
            outputs=[wp.from_torch(result, dtype=wp.bool)],
            stream=wp.stream_from_torch(torch.cuda.current_stream(start.device)),
        )
    return result.all(-1)
