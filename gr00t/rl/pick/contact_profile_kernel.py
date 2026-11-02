"""Fused GPU inner-jaw clearance, without expanding points over closing angles."""

import torch
import warp as wp

BLOCK_DIM = wp.constant(128)
MAX_CONTACT_POINTS = 8192


@wp.func
def _sample(profile: wp.array3d(dtype=float), channel: int, y: int, x: int):
    value = float(0.0)
    if y >= 0 and y < profile.shape[1] and x >= 0 and x < profile.shape[2]:
        value = profile[channel, y, x]
    return value


@wp.func
def _bilinear(profile: wp.array3d(dtype=float), channel: int, y: float, x: float):
    ix, iy = int(wp.floor(x)), int(wp.floor(y))
    dx, dy = x - float(ix), y - float(iy)
    return (
        _sample(profile, channel, iy, ix) * (1.0 - dx) * (1.0 - dy)
        + _sample(profile, channel, iy, ix + 1) * dx * (1.0 - dy)
        + _sample(profile, channel, iy + 1, ix) * (1.0 - dx) * dy
        + _sample(profile, channel, iy + 1, ix + 1) * dx * dy
    )


@wp.kernel
def _minimum_gap(
    points: wp.array2d(dtype=wp.vec3),
    centers: wp.array(dtype=wp.vec3),
    angles: wp.array(dtype=float),
    profile: wp.array3d(dtype=float),
    depth_count: int,
    angle_count: int,
    output: wp.array(dtype=float),
):
    row, chunk, lane = wp.tid()
    sample = chunk * BLOCK_DIM + lane
    center_row = row // angle_count
    orientation_row = center_row // depth_count
    gap = float(wp.inf)
    if sample < points.shape[1]:
        point = points[orientation_row, sample] + centers[center_row]
        c, s = wp.cos(angles[row]), wp.sin(angles[row])
        x, z = point[0] * c + point[2] * s, -point[0] * s + point[2] * c
        gx = ((x - 0.01) / 0.11 + 1.0) * 0.5 * float(profile.shape[2] - 1)
        gy = (point[1] / 0.05 + 1.0) * 0.5 * float(profile.shape[1] - 1)
        if _bilinear(profile, 4, gy, gx) > 0.99:
            gap = _bilinear(profile, 0, gy, gx) - z
    minimum = wp.tile_min(wp.tile(gap))[0]
    if lane == 0:
        wp.atomic_min(output, row, minimum)


def moving_profile_minimum(local, centers, angles, profile):
    """Match grid_sample(..., align_corners=True, padding_mode='zeros') then min."""
    if local.device.type != "cuda" or local.dtype != torch.float32:
        raise ValueError("Fused contact profiles require CUDA float32 tensors")
    if local.ndim != 4 or centers.ndim != 4 or angles.ndim not in (3, 4):
        raise ValueError("Expected local[B,K,N,3], centers[B,K,D,3], angles[B,K,D,(A)]")
    if local.shape[-2] > MAX_CONTACT_POINTS:
        result = angles.new_full(angles.shape, torch.inf)
        for chunk in local.split(MAX_CONTACT_POINTS, dim=-2):
            result = torch.minimum(result, moving_profile_minimum(chunk, centers, angles, profile))
        return result
    wp.init()
    angle_count = 1 if angles.ndim == 3 else angles.shape[-1]
    result = angles.new_full(angles.shape, torch.inf)
    if not angles.numel() or not local.shape[-2]:
        return result
    points = local.flatten(0, 1).contiguous()
    center = centers.flatten(0, 2).contiguous()
    angle = angles.contiguous().flatten()
    wp.launch_tiled(
        _minimum_gap,
        dim=(angle.numel(), (local.shape[-2] + BLOCK_DIM - 1) // BLOCK_DIM),
        inputs=[
            wp.from_torch(points, dtype=wp.vec3),
            wp.from_torch(center, dtype=wp.vec3),
            wp.from_torch(angle),
            wp.from_torch(profile[0].contiguous()),
            centers.shape[-2],
            angle_count,
            wp.from_torch(result.flatten()),
        ],
        stream=wp.stream_from_torch(torch.cuda.current_stream(local.device)),
        block_dim=BLOCK_DIM,
    )
    return result


@wp.func
def _point_gap(point: wp.vec3, angle: float, profile: wp.array3d(dtype=float)):
    c, s = wp.cos(angle), wp.sin(angle)
    x, z = point[0] * c + point[2] * s, -point[0] * s + point[2] * c
    gx = ((x - 0.01) / 0.11 + 1.0) * 0.5 * float(profile.shape[2] - 1)
    gy = (point[1] / 0.05 + 1.0) * 0.5 * float(profile.shape[1] - 1)
    gap = float(wp.inf)
    if _bilinear(profile, 4, gy, gx) > 0.99:
        gap = _bilinear(profile, 0, gy, gx) - z
    return gap


@wp.kernel
def _closing_contact(
    points: wp.array2d(dtype=wp.vec3),
    centers: wp.array(dtype=wp.vec3),
    open_angles: wp.array(dtype=float),
    profile: wp.array3d(dtype=float),
    depth_count: int,
    output: wp.array(dtype=float),
):
    row, lane = wp.tid()
    orientation = row // depth_count
    last_collision = int(0)
    for step in range(17):
        angle = open_angles[row] * (float(step) / 16.0)
        gap = float(wp.inf)
        sample = lane
        # Only the sign enters the bracket; a collision cannot become nonnegative.
        while sample < points.shape[1] and gap >= 0.0:
            gap = wp.min(
                gap, _point_gap(points[orientation, sample] + centers[row], angle, profile)
            )
            sample += BLOCK_DIM
        minimum = wp.tile_min(wp.tile(gap))[0]
        if minimum < 0.0:
            last_collision = step
    lower = open_angles[row] * (float(last_collision) / 16.0)
    upper = open_angles[row] * (float(wp.min(last_collision + 1, 16)) / 16.0)
    for iteration in range(16):
        angle = (lower + upper) / 2.0
        gap = float(wp.inf)
        sample = lane
        while sample < points.shape[1] and gap >= 0.0:
            gap = wp.min(
                gap, _point_gap(points[orientation, sample] + centers[row], angle, profile)
            )
            sample += BLOCK_DIM
        minimum = wp.tile_min(wp.tile(gap))[0]
        if minimum < 0.0:
            lower = angle
        else:
            upper = angle
    if lane == 0:
        output[row] = (lower + upper) / 2.0


def first_profile_contact(local, centers, open_angles, profile):
    """Same 17-sample bracket and 16 bisections, in one GPU launch."""
    if local.device.type != "cuda" or local.dtype != torch.float32:
        raise ValueError("Fused closing contact requires CUDA float32 tensors")
    if local.ndim != 4 or centers.ndim != 4 or open_angles.shape != centers.shape[:-1]:
        raise ValueError("Expected local[B,K,N,3], centers[B,K,D,3], angles[B,K,D]")
    if local.shape[-2] > MAX_CONTACT_POINTS:
        sweep = torch.linspace(0, 1, 17, device=local.device, dtype=local.dtype)
        gaps = moving_profile_minimum(local, centers, open_angles[..., None] * sweep, profile)
        last = torch.where(gaps < 0, torch.arange(17, device=local.device), 0).amax(-1)
        lower = open_angles * sweep[last]
        upper = open_angles * sweep[(last + 1).clamp_max(16)]
        for _ in range(16):
            angle = (lower + upper) / 2
            gap = moving_profile_minimum(local, centers, angle, profile)
            lower = torch.where(gap < 0, angle, lower)
            upper = torch.where(gap >= 0, angle, upper)
        return (lower + upper) / 2
    wp.init()
    result = torch.empty_like(open_angles)
    if not result.numel():
        return result
    wp.launch_tiled(
        _closing_contact,
        dim=(result.numel(),),
        inputs=[
            wp.from_torch(local.flatten(0, 1).contiguous(), dtype=wp.vec3),
            wp.from_torch(centers.flatten(0, 2).contiguous(), dtype=wp.vec3),
            wp.from_torch(open_angles.contiguous().flatten()),
            wp.from_torch(profile[0].contiguous()),
            centers.shape[-2],
            wp.from_torch(result.flatten()),
        ],
        stream=wp.stream_from_torch(torch.cuda.current_stream(local.device)),
        block_dim=BLOCK_DIM,
    )
    return result


@wp.kernel
def _fixed_centers(
    points: wp.array2d(dtype=wp.vec3),
    centers: wp.array(dtype=wp.vec3),
    profile: wp.array3d(dtype=float),
    depths: int,
    output: wp.array(dtype=wp.vec3),
    valid: wp.array(dtype=bool),
):
    row, lane = wp.tid()
    gap = float(-wp.inf)
    center = centers[row]
    for sample in range(lane, points.shape[1], BLOCK_DIM):
        value = _point_gap(points[row // depths, sample] + center, 0.0, profile)
        if value < wp.inf:
            gap = wp.max(gap, value)
    required = wp.tile_max(wp.tile(gap))[0]
    if lane == 0:
        valid[row] = required > -wp.inf
        if required > -wp.inf:
            center[2] = required
        output[row] = center


@wp.func
def _surface_normal(point: wp.vec3, angle: float, profile: wp.array3d(dtype=float)):
    c, s = wp.cos(angle), wp.sin(angle)
    x = point[0] * c + point[2] * s
    gx = ((x - 0.01) / 0.11 + 1.0) * 0.5 * float(profile.shape[2] - 1)
    gy = (point[1] / 0.05 + 1.0) * 0.5 * float(profile.shape[1] - 1)
    n = wp.vec3(
        _bilinear(profile, 1, gy, gx), _bilinear(profile, 2, gy, gx), _bilinear(profile, 3, gy, gx)
    )
    n = n / wp.max(wp.length(n), 1e-8)
    return wp.vec3(c * n[0] - s * n[2], n[1], s * n[0] + c * n[2])


@wp.kernel
def _patch_statistics(
    points: wp.array2d(dtype=wp.vec3),
    centers: wp.array(dtype=wp.vec3),
    angles: wp.array(dtype=float),
    mask: wp.array2d(dtype=bool),
    fixed: wp.array3d(dtype=float),
    moving: wp.array3d(dtype=float),
    depths: int,
    candidates: int,
    external_gaps: wp.array2d(dtype=float),
    use_external_gaps: bool,
    output: wp.array3d(dtype=wp.float64),
    gaps: wp.array2d(dtype=float),
):
    row, lane = wp.tid()
    orientation = row // depths
    owner = orientation // candidates
    center, angle = centers[row], angles[row]
    fg, mg = float(wp.inf), float(wp.inf)
    for sample in range(lane, points.shape[1], BLOCK_DIM):
        if mask[owner, sample]:
            point = points[orientation, sample] + center
            gap = _point_gap(point, 0.0, fixed)
            if gap < wp.inf:
                fg = wp.min(fg, -gap)
            mg = wp.min(mg, _point_gap(point, angle, moving))
    fmin = wp.tile_min(wp.tile(fg))[0]
    mmin = wp.tile_min(wp.tile(mg))[0]
    if use_external_gaps:
        fmin = external_gaps[row, 0]
        mmin = external_gaps[row, 1]
    if lane == 0:
        gaps[row, 0] = fmin
        gaps[row, 1] = mmin
    # One deterministic block reduction per pose; no point-by-depth materialization.
    for jaw in range(2):
        px, py, pz = wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)
        nx, ny, nz = wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)
        count = wp.float64(0.0)
        for sample in range(lane, points.shape[1], BLOCK_DIM):
            if mask[owner, sample]:
                point = points[orientation, sample] + center
                gap = _point_gap(point, angle, moving)
                patch = gap < wp.inf and gap <= mmin + 0.0003
                normal = wp.vec3(0.0)
                if jaw == 0:
                    value = _point_gap(point, 0.0, fixed)
                    patch = value < wp.inf and -value <= fmin + 0.0003
                    if patch:
                        normal = _surface_normal(point, 0.0, fixed)
                elif patch:
                    normal = _surface_normal(point, angle, moving)
                if patch:
                    px += wp.float64(point[0])
                    py += wp.float64(point[1])
                    pz += wp.float64(point[2])
                    nx += wp.float64(normal[0])
                    ny += wp.float64(normal[1])
                    nz += wp.float64(normal[2])
                    count += wp.float64(1.0)
        sx = wp.tile_sum(wp.tile(px))[0]
        sy = wp.tile_sum(wp.tile(py))[0]
        sz = wp.tile_sum(wp.tile(pz))[0]
        snx = wp.tile_sum(wp.tile(nx))[0]
        sny = wp.tile_sum(wp.tile(ny))[0]
        snz = wp.tile_sum(wp.tile(nz))[0]
        sc = wp.tile_sum(wp.tile(count))[0]
        if lane == 0:
            output[row, jaw, 0] = sx
            output[row, jaw, 1] = sy
            output[row, jaw, 2] = sz
            output[row, jaw, 3] = snx
            output[row, jaw, 4] = sny
            output[row, jaw, 5] = snz
            output[row, jaw, 6] = sc


def fixed_profile_centers(local, centers, profile):
    if local.shape[-2] > MAX_CONTACT_POINTS:
        required = centers.new_full(centers.shape[:-1], -torch.inf)
        for chunk in local.split(MAX_CONTACT_POINTS, dim=-2):
            partial, valid = fixed_profile_centers(chunk, centers, profile)
            required = torch.maximum(required, partial[..., 2].masked_fill(~valid, -torch.inf))
        valid = torch.isfinite(required)
        result = centers.clone()
        result[..., 2] = torch.where(valid, required, centers[..., 2])
        return result, valid
    wp.init()
    result = torch.empty_like(centers)
    valid = torch.empty(centers.shape[:-1], device=local.device, dtype=torch.bool)
    wp.launch_tiled(
        _fixed_centers,
        dim=(centers.numel() // 3,),
        inputs=[
            wp.from_torch(local.flatten(0, 1).contiguous(), dtype=wp.vec3),
            wp.from_torch(centers.flatten(0, 2).contiguous(), dtype=wp.vec3),
            wp.from_torch(profile[0].contiguous()),
            centers.shape[-2],
            wp.from_torch(result.flatten(0, 2), dtype=wp.vec3),
            wp.from_torch(valid.flatten()),
        ],
        stream=wp.stream_from_torch(torch.cuda.current_stream(local.device)),
        block_dim=BLOCK_DIM,
    )
    return result, valid


def _profile_statistics_values(
    local, centers, angles, fixed, moving, support_mask, external_gaps=None
):
    wp.init()
    values = torch.empty((angles.numel(), 2, 7), device=local.device, dtype=torch.float64)
    gaps = angles.new_empty(angles.numel(), 2)
    if support_mask is None:
        support_mask = torch.ones(
            (local.shape[0], local.shape[2]), device=local.device, dtype=torch.bool
        )
    supplied_gaps = gaps if external_gaps is None else external_gaps.reshape(-1, 2).contiguous()
    wp.launch_tiled(
        _patch_statistics,
        dim=(angles.numel(),),
        inputs=[
            wp.from_torch(local.flatten(0, 1).contiguous(), dtype=wp.vec3),
            wp.from_torch(centers.flatten(0, 2).contiguous(), dtype=wp.vec3),
            wp.from_torch(angles.flatten().contiguous()),
            wp.from_torch(support_mask.contiguous()),
            wp.from_torch(fixed[0].contiguous()),
            wp.from_torch(moving[0].contiguous()),
            centers.shape[-2],
            local.shape[1],
            wp.from_torch(supplied_gaps),
            external_gaps is not None,
            wp.from_torch(values),
            wp.from_torch(gaps),
        ],
        stream=wp.stream_from_torch(torch.cuda.current_stream(local.device)),
        block_dim=BLOCK_DIM,
    )
    return values, gaps


def profile_contact_statistics(local, centers, angles, fixed, moving, support_mask=None):
    shape = centers.shape[:-1]
    if local.shape[-2] > MAX_CONTACT_POINTS:
        # Global patch minima must be shared by all chunks; averaging contacts
        # from independently thresholded chunks would change the grasp geometry.
        if support_mask is None:
            support_mask = torch.ones(
                (local.shape[0], local.shape[-2]), device=local.device, dtype=torch.bool
            )
        masked_local = local.masked_fill(~support_mask[:, None, :, None], 1e5)
        fcenters, valid = fixed_profile_centers(masked_local, centers, fixed)
        fg = (-fcenters[..., 2]).masked_fill(~valid, torch.inf)
        mg = moving_profile_minimum(masked_local, centers, angles, moving)
        gaps = torch.stack((fg, mg), -1)
        values = torch.zeros((angles.numel(), 2, 7), device=local.device, dtype=torch.float64)
        for start in range(0, local.shape[-2], MAX_CONTACT_POINTS):
            chunk = local[..., start : start + MAX_CONTACT_POINTS, :]
            mask = support_mask[:, start : start + MAX_CONTACT_POINTS]
            partial, _ = _profile_statistics_values(
                chunk, centers, angles, fixed, moving, mask, gaps
            )
            values += partial
    else:
        values, gaps = _profile_statistics_values(
            local, centers, angles, fixed, moving, support_mask
        )
    contacts = (values[..., :3] / values[..., 6:7].clamp_min(1)).float()
    normals = values[..., 3:6]
    normals = (normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-8)).float()
    return contacts.reshape(*shape, 2, 3), normals.reshape(*shape, 2, 3), gaps.reshape(*shape, 2)
