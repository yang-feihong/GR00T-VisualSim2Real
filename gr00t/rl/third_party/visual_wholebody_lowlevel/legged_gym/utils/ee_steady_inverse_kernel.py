"""Fused, row-independent CUDA implementation of the steady inverse iteration."""

import warp as wp

from gr00t.rl.utils.b2z1_asset import read_arm_mount

from gr00t.rl.envs.door.b2z1_reset_geometry import (
    _ARM_JOINT_AXES,
    _ARM_JOINT_ORIGINS,
)

wp.init()
wp.set_module_options({"enable_backward": False, "fast_math": False})
V6 = wp.types.vector(length=6, dtype=wp.float64)
M6 = wp.types.matrix(shape=(6, 6), dtype=wp.float64)
M63 = wp.types.matrix(shape=(6, 3), dtype=wp.float64)
ORIGINS = wp.constant(M63(*[v for row in _ARM_JOINT_ORIGINS for v in row]))
AXES = wp.constant(wp.types.vector(length=6, dtype=wp.int32)(*_ARM_JOINT_AXES))
ZERO = wp.constant(wp.float64(0.0))
ONE = wp.constant(wp.float64(1.0))
MOUNT = wp.constant(wp.vec3d(*read_arm_mount()[0]))
MOUNT_YAW = wp.constant(wp.float64(read_arm_mount()[1]))


@wp.func
def _rotation(axis: int, angle: wp.float64):
    c, s = wp.cos(angle), wp.sin(angle)
    r = wp.mat33d(wp.float64(1.0))
    if axis == 0:
        r = wp.mat33d(ONE, ZERO, ZERO, ZERO, c, -s, ZERO, s, c)
    elif axis == 1:
        r = wp.mat33d(c, ZERO, s, ZERO, ONE, ZERO, -s, ZERO, c)
    else:
        r = wp.mat33d(c, -s, ZERO, s, c, ZERO, ZERO, ZERO, ONE)
    return r


@wp.func
def _fk(base: wp.vec3d, base_q: wp.quatd, arm: V6):
    rotation = wp.quat_to_matrix(base_q)
    position = base + rotation * MOUNT
    rotation = rotation * _rotation(2, MOUNT_YAW)
    origins, axes = M63(), M63()
    for joint in range(6):
        position = position + rotation * wp.vec3d(
            ORIGINS[joint, 0], ORIGINS[joint, 1], ORIGINS[joint, 2]
        )
        for row in range(3):
            origins[joint, row] = position[row]
            axes[joint, row] = rotation[row, AXES[joint]]
        rotation = rotation * _rotation(AXES[joint], arm[joint])
    position = position + rotation * wp.vec3d(wp.float64(0.051), ZERO, ZERO)
    rotation = rotation * _rotation(0, wp.float64(-1.5707963267948966))
    position = position + rotation * wp.vec3d(wp.float64(0.049), ZERO, ZERO)
    jacobian = M6()
    for joint in range(6):
        axis = wp.vec3d(axes[joint, 0], axes[joint, 1], axes[joint, 2])
        origin = wp.vec3d(origins[joint, 0], origins[joint, 1], origins[joint, 2])
        linear = wp.cross(axis, position - origin)
        for row in range(3):
            jacobian[row, joint] = linear[row]
            jacobian[row + 3, joint] = axis[row]
    return position, wp.normalize(wp.quat_from_matrix(rotation)), jacobian


@wp.func
def _orientation_error(desired: wp.quatd, current: wp.quatd):
    relative = desired * wp.quat_inverse(current)
    if relative[3] < wp.float64(0.0):
        relative = -relative
    return relative


@wp.func
def _spd_solve(matrix: M6, rhs: V6):
    lower = M6()
    for row in range(6):
        for column in range(row + 1):
            value = matrix[row, column]
            for inner in range(column):
                value = value - lower[row, inner] * lower[column, inner]
            if row == column:
                lower[row, column] = wp.sqrt(wp.max(value, wp.float64(1.0e-12)))
            else:
                lower[row, column] = value / lower[column, column]
    forward = V6()
    for row in range(6):
        value = rhs[row]
        for column in range(row):
            value = value - lower[row, column] * forward[column]
        forward[row] = value / lower[row, row]
    solution = V6()
    for index in range(6):
        row = 5 - index
        value = forward[row]
        for column in range(row + 1, 6):
            value = value - lower[column, row] * solution[column]
        solution[row] = value / lower[row, row]
    return solution


@wp.func
def _constrained_delta(jacobian: M6, error: V6, arm: V6,
                       lower: V6, upper: V6, damping: wp.float64):
    delta, free = V6(), V6(wp.float64(1.0))
    for iteration in range(7):
        residual = error
        for joint in range(6):
            if free[joint] == wp.float64(0.0):
                for row in range(6):
                    residual[row] = residual[row] - jacobian[row, joint] * delta[joint]
        normal, rhs = M6(), V6()
        for row in range(6):
            for i in range(6):
                ji = jacobian[row, i] * free[i]
                rhs[i] = rhs[i] + ji * residual[row]
                for j in range(6):
                    normal[i, j] = normal[i, j] + ji * jacobian[row, j] * free[j]
        for joint in range(6):
            normal[joint, joint] = normal[joint, joint] + (
                free[joint] * damping * damping + wp.float64(1.0) - free[joint]
            )
        candidate = _spd_solve(normal, rhs)
        violation = False
        for joint in range(6):
            if free[joint] > wp.float64(0.0):
                lo = lower[joint] + wp.float64(0.001) - arm[joint]
                hi = upper[joint] - wp.float64(0.001) - arm[joint]
                if candidate[joint] < lo or candidate[joint] > hi:
                    violation = True
        if not violation:
            for joint in range(6):
                if free[joint] > wp.float64(0.0):
                    delta[joint] = candidate[joint]
            break
        for joint in range(6):
            if free[joint] > wp.float64(0.0):
                lo = lower[joint] + wp.float64(0.001) - arm[joint]
                hi = upper[joint] - wp.float64(0.001) - arm[joint]
                if candidate[joint] < lo or candidate[joint] > hi:
                    delta[joint] = wp.clamp(candidate[joint], lo, hi)
                    free[joint] = wp.float64(0.0)
    return delta


@wp.kernel
def steady_inverse_kernel(
    positions: wp.array2d(dtype=wp.float64),
    orientations: wp.array2d(dtype=wp.float64),
    roots: wp.array2d(dtype=wp.float64),
    root_quaternions: wp.array2d(dtype=wp.float64),
    initial: wp.array2d(dtype=wp.float64),
    limits: wp.array3d(dtype=wp.float64),
    position_step: wp.float64,
    orientation_step: wp.float64,
    damping: wp.float64,
    max_iterations: int,
    solved: wp.array2d(dtype=wp.float64),
    iterations: wp.array(dtype=wp.int32),
):
    row = wp.tid()
    base = wp.vec3d(roots[row, 0], roots[row, 1], roots[row, 2])
    base_q = wp.normalize(wp.quatd(root_quaternions[row, 0], root_quaternions[row, 1],
                                  root_quaternions[row, 2], root_quaternions[row, 3]))
    desired = wp.vec3d(positions[row, 0], positions[row, 1], positions[row, 2])
    desired_q = wp.normalize(wp.quatd(orientations[row, 1], orientations[row, 2],
                                     orientations[row, 3], orientations[row, 0]))
    arm, lower, upper = V6(), V6(), V6()
    for joint in range(6):
        arm[joint] = initial[row, joint]
        lower[joint] = limits[row, joint, 0]
        upper[joint] = limits[row, joint, 1]
    start, start_q, jacobian = _fk(base, base_q, arm)
    dot = wp.dot(start_q, desired_q)
    if dot < wp.float64(0.0):
        desired_q = -desired_q
    distance = wp.float64(2.0) * wp.acos(wp.clamp(wp.abs(dot), wp.float64(0.0), wp.float64(1.0)))
    count = wp.max(1, wp.max(int(wp.ceil(wp.length(desired - start) / position_step)),
                             int(wp.ceil(distance / orientation_step))))
    used = int(0)
    for waypoint in range(1, count + 1):
        fraction = wp.float64(waypoint) / wp.float64(count)
        point = start + fraction * (desired - start)
        quat = wp.normalize(start_q + fraction * (desired_q - start_q))
        for iteration in range(max_iterations):
            position, orientation, jacobian = _fk(base, base_q, arm)
            rotation_error = _orientation_error(quat, orientation)
            offset = point - position
            # Solve tolerance is deliberately tighter than every caller's acceptance.
            if wp.length(offset) <= wp.float64(1.0e-10) and wp.length(wp.vec3d(
                rotation_error[0], rotation_error[1], rotation_error[2]
            )) <= wp.float64(1.0e-10):
                break
            error = V6()
            vector = wp.vec3d(rotation_error[0], rotation_error[1], rotation_error[2])
            skew = wp.mat33d(ZERO, -vector[2], vector[1], vector[2], ZERO,
                            -vector[0], -vector[1], vector[0], ZERO)
            weighted = M6()
            for i in range(3):
                error[i] = wp.float64(2.5) * offset[i]
                error[i + 3] = vector[i]
                for j in range(6):
                    weighted[i, j] = wp.float64(2.5) * jacobian[i, j]
                    value = rotation_error[3] * jacobian[i + 3, j]
                    for k in range(3):
                        value = value + skew[i, k] * jacobian[k + 3, j]
                    weighted[i + 3, j] = wp.float64(0.5) * value
            delta = _constrained_delta(weighted, error, arm, lower, upper, damping)
            arm = arm + delta
            used += 1
            # Bound-constrained stationary points need not have zero pose residual.
            if wp.length(delta) <= wp.float64(1.0e-12):
                break
    for joint in range(6):
        solved[row, joint] = arm[joint]
    iterations[row] = used
