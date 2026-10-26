"""Device-only shared stage planning for the tensor FSM.

Put GPUStagePlanningMixin before the coordinator base in the MRO and call
initialize_gpu_stage_constants() once after setting controllers. The host owns
_inverse_pose_bank(fsm, state, rows, positions, orientations, lengths),
_record_stats(label, requests, errors=None), and path_diagnostics. Runtime stage
methods never read controllers or materialize tensor values on the host.
"""

from __future__ import annotations

import math

import torch
from gr00t.rl.auto_open.geometry import _quat_rotate_points_torch

from gr00t.rl.auto_open.tensor_fsm import (
    _axis_angle_quat,
    _euler_xyz_quat,
    _quat_multiply,
    _quat_conjugate,
    _yaw_quat,
)


def _quat_rotate(quaternion, vector):
    q_vector, vector = torch.broadcast_tensors(quaternion[..., 1:], vector)
    cross = torch.cross(q_vector, vector, dim=-1)
    return vector + 2 * (quaternion[..., :1] * cross + torch.cross(q_vector, cross, dim=-1))


def _normalize(q):
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1.e-12)


def panel_forward_anchor_distance(root, hinge, panel_tangent, forward_offset):
    """Project the actual base's yaw-forward point onto the panel hinge radius."""
    w, x, y, z = root[:, 3:7].unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    forward = torch.stack((yaw.cos(), yaw.sin()), -1)
    relative = root[:, :2] + forward_offset * forward - hinge[:, :2]
    return relative[:, 0] * panel_tangent[:, 0] + relative[:, 1] * panel_tangent[:, 1]


def panel_handle_height_mask(candidate_height, jaw_z_bounds, handle_z_bounds,
                             clearance_m=.01):
    """Keep the entire jaw height interval outside the handle band."""
    lower = candidate_height + jaw_z_bounds[:, :1]
    upper = candidate_height + jaw_z_bounds[:, 1:2]
    return ((lower >= handle_z_bounds[:, 1:2] + clearance_m)
            | (upper <= handle_z_bounds[:, :1] - clearance_m))


def _nlerp(start, end, fraction):
    end = torch.where((start * end).sum(-1, keepdim=True) < 0, -end, end)
    return _normalize(start + fraction[..., None] * (end - start))


def _slerp(start, end, fraction):
    start, end = _normalize(start), _normalize(end)
    end = torch.where((start * end).sum(-1, keepdim=True) < 0, -end, end)
    half_angle = torch.acos((start * end).sum(-1).clamp(-1., 1.))
    sine = half_angle.sin().clamp_min(1.e-12)
    result = (torch.sin((1 - fraction) * half_angle)[..., None] * start
              + torch.sin(fraction * half_angle)[..., None] * end) / sine[..., None]
    return torch.where((half_angle < 5.e-9)[..., None], start, result)


def path_samples(start, end, start_q, end_q, capacity, *, steps=None):
    """Match legacy steady (excluding start) and release (including start) paths.

    Return padded poses and the *unclipped* lengths. Callers must reject lengths
    exceeding capacity, never silently truncate a requested motion.
    """
    distance = torch.linalg.vector_norm(end - start, dim=-1)
    if steps is None:
        angle = 2 * torch.acos((start_q * end_q).sum(-1).abs().clamp(0., 1.))
        count = torch.maximum(torch.ceil(distance / .002), torch.ceil(angle / .03))
        count = count.clamp_min(1).to(torch.int64)
        length = count
        sample = torch.arange(1, capacity + 1, device=start.device)
    else:
        count = steps.to(torch.int64)
        length = count + 1
        sample = torch.arange(capacity, device=start.device)
    fraction = (sample[None] / count[:, None]).clamp_max(1.)
    positions = start[:, None] + fraction[..., None] * (end - start)[:, None]
    orientations = _nlerp(start_q[:, None], end_q[:, None], fraction)
    return positions, orientations, length


def pregrasp_avoidance_waypoints(state, geometry, rows, normal, end, end_q,
                                 *, height_clearance_m=.002, retreat_clearance_m=.02):
    """Separate vertically, withdraw normally, then approach the unchanged goal."""
    start, start_q = state['ee_pos'][rows].double(), state['ee_orn'][rows].double()
    bodies = state['robot_body_state_w'][rows].double()
    fixed_pose = bodies[:, geometry['fixed_body_idx'], :7]
    moving_pose = bodies[:, geometry['moving_body_idx'], :7]
    fixed = _quat_rotate_points_torch(fixed_pose[:, 3:7], geometry['fixed_points'].double()) + fixed_pose[:, None, :3]
    moving = _quat_rotate_points_torch(moving_pose[:, 3:7], geometry['moving_points'].double()) + moving_pose[:, None, :3]
    lower = torch.minimum(fixed[..., 2].amin(1), moving[..., 2].amin(1))
    upper = torch.maximum(fixed[..., 2].amax(1), moving[..., 2].amax(1))
    handle = state['door_body_state_w'][rows, geometry['handle_body_idx'], :7].double()
    center = handle[:, :3] + _quat_rotate(handle[:, 3:7], geometry['handle_center_local'][rows].double())
    axis = _quat_rotate(handle[:, 3:7], geometry['handle_axis_local'][rows].double())
    half, radius = geometry['handle_half_length'][rows].double(), geometry['handle_radius'][rows].double()
    extent_z = half * axis[:, 2].abs() + radius
    handle_lower, handle_upper = center[:, 2] - extent_z, center[:, 2] + extent_z
    above = (lower + upper) * .5 >= center[:, 2]
    shift = torch.where(above, (handle_upper + height_clearance_m - lower).clamp_min(0.),
                        (handle_lower - height_clearance_m - upper).clamp_max(0.))
    raised = start.clone()
    raised[:, 2] += shift
    normal = normal.double()
    normal_extent = half * (axis * normal).sum(-1).abs() + radius
    handle_front = (center * normal).sum(-1) + normal_extent
    jaw_front = torch.minimum((fixed * normal[:, None]).sum(-1).amin(1),
                              (moving * normal[:, None]).sum(-1).amin(1))
    retreat = (handle_front + retreat_clearance_m - jaw_front).clamp_min(0.)
    withdrawn = raised + retreat[:, None] * normal
    points = torch.stack((start, raised, withdrawn, end.double()), 1)
    quats = torch.stack((start_q, start_q, start_q, end_q.double()), 1)
    diagnostics = dict(waypoints_world_m=points, waypoint_orientations_wxyz=quats,
        vertical_shift_m=shift, normal_retreat_m=retreat,
        initial_gripper_lower_z_m=lower, initial_gripper_upper_z_m=upper,
        handle_lower_z_m=handle_lower, handle_upper_z_m=handle_upper,
        height_clearance_after_shift_m=torch.where(above, lower + shift - handle_upper,
                                                   handle_lower - upper - shift),
        normal_clearance_after_retreat_m=jaw_front + retreat - handle_front)
    return points, quats, diagnostics


def _segment_indices(count):
    """Pack ragged inclusive samples without copying a maximum to the host."""
    length = count + 1
    owner = torch.repeat_interleave(torch.arange(len(count), device=count.device), length)
    offset = torch.cumsum(length, 0) - length
    sample = torch.arange(owner.shape[0], device=count.device) - offset[owner]
    return owner, sample / count[owner]


def pregrasp_pose_family(start, start_q, grasp, grasp_q, axle, axis, hinge,
                         hinge_angle, handle_start, handle_target):
    """Pack insertion, press, and three door/return branches for each candidate.

    Input leading dimension is the flattened (environment, pitch) dimension.
    Returns flat poses plus candidate owners, in legacy sample order per owner.
    Duplicate branches when release == target are harmless for exact acceptance.
    """
    start_q, grasp_q = _normalize(start_q), _normalize(grasp_q)
    angle = 2 * torch.acos((start_q * grasp_q).sum(-1).abs().clamp(0., 1.))
    count = torch.maximum(torch.ceil(torch.linalg.vector_norm(grasp - start, dim=-1) / .004),
                          torch.ceil(angle / math.radians(1))).clamp_min(1).long()
    owner, fraction = _segment_indices(count)
    positions = [start[owner] + fraction[:, None] * (grasp - start)[owner]]
    orientations = [_slerp(start_q[owner], grasp_q[owner], fraction)]
    owners = [owner]
    axis = _normalize(axis)

    def rotation_segment(point, quat, pivot, rotation_axis, rotation_angle):
        radius = torch.linalg.vector_norm(torch.cross(point - pivot, rotation_axis, dim=-1), dim=-1)
        count = torch.maximum(torch.ceil(rotation_angle.abs() / math.radians(1)),
                              torch.ceil(rotation_angle.abs() * radius / .004)).clamp_min(1).long()
        owner, fraction = _segment_indices(count)
        rotation = _axis_angle_quat(rotation_axis[owner], rotation_angle[owner] * fraction)
        positions.append(pivot[owner] + _quat_rotate(rotation, (point - pivot)[owner]))
        orientations.append(_quat_multiply(rotation, quat[owner]))
        owners.append(owner)
        final = _axis_angle_quat(rotation_axis, rotation_angle)
        return pivot + _quat_rotate(final, point - pivot), _quat_multiply(final, quat)

    target = torch.maximum(handle_start, handle_target)
    rotation_segment(grasp, grasp_q, axle, axis, target - handle_start)
    release = torch.minimum(torch.maximum(torch.full_like(target, math.radians(45) * .025 / .03),
                                          handle_start), target)
    z_axis = torch.zeros_like(axis)
    z_axis[:, 2] = 1
    door_rotation = _axis_angle_quat(z_axis, hinge_angle)
    return_axle = hinge + _quat_rotate(door_rotation, axle - hinge)
    return_axis = _quat_rotate(door_rotation, axis)
    for branch in (release, (release + target) / 2, target):
        rotation = _axis_angle_quat(axis, branch - handle_start)
        point = axle + _quat_rotate(rotation, grasp - axle)
        quat = _quat_multiply(rotation, grasp_q)
        point, quat = rotation_segment(point, quat, hinge, z_axis, hinge_angle)
        rotation_segment(point, quat, return_axle, return_axis, handle_start - branch)
    # Stable owner sort retains segment and sample order for every candidate.
    owner = torch.cat(owners)
    order = torch.argsort(owner, stable=True)
    return torch.cat(positions)[order], torch.cat(orientations)[order], owner[order]


def center_vertical_shift(error, fixed_gap):
    needed = (error.abs() - .008).clamp_min(0.)
    available = (.019 - fixed_gap.clamp_min(0.)).clamp_min(0.)
    shift = -torch.copysign(torch.minimum(needed, available), error)
    return shift, (error + shift).abs() <= .010


def align_vectors(source, target, base_quat):
    source, target = _normalize(source), _normalize(target)
    cross = torch.cross(source, target, dim=-1)
    dot = (source * target).sum(-1).clamp(-1., 1.)
    x_axis = torch.zeros_like(source)
    x_axis[:, 0] = 1
    y_axis = torch.zeros_like(source)
    y_axis[:, 1] = 1
    axis = torch.cross(source, x_axis, dim=-1)
    axis = torch.where((torch.linalg.vector_norm(axis, dim=-1) < 1.e-5)[:, None],
                       torch.cross(source, y_axis, dim=-1), axis)
    opposite = torch.cat((torch.zeros_like(dot[:, None]), _normalize(axis)), -1)
    delta = _normalize(torch.cat(((1 + dot)[:, None], cross), -1))
    return _quat_multiply(torch.where((dot < -.9999)[:, None], opposite, delta), base_quat)


def grasp_noise_target(cache, nominal, nominal_q):
    """Apply the shared grasp-chain sample and asymmetric axial jaw margins."""
    depth = _normalize(_quat_rotate(nominal_q, cache["depth_axis"][:, None]))
    closing = _quat_rotate(nominal_q, cache["separation_axis"][:, None])
    closing = _normalize(closing - depth * (closing * depth).sum(-1, keepdim=True))
    handle_axis = _normalize(torch.cross(closing, depth, dim=-1))
    sample = cache["grasp_position_noise"][:, None]
    raw_delta = sample[..., :1] * depth + sample[..., 1:2] * handle_axis + sample[..., 2:] * closing
    rotation = _axis_angle_quat(handle_axis.flatten(0, 1),
        cache["grasp_rotation_noise"][:, :1].expand(-1, nominal.shape[1]).flatten())
    quat = _quat_multiply(rotation.reshape_as(nominal_q), nominal_q)
    target = nominal + raw_delta
    jaw_offsets = _quat_rotate(quat[:, :, None], cache["jaw_points"][:, None])
    free_axis = cache["handle_free_axis_world"][:, None]
    axial = (jaw_offsets * free_axis[:, :, None]).sum(-1)
    axial_min, axial_max = axial.amin(-1), axial.amax(-1)
    center_offset = ((target - cache["grasp_center_pos"][:, None]) * free_axis).sum(-1)
    outer = cache["outer_half_length"][:, None]
    low, high = -outer + .025 - center_offset - axial_min, outer - .015 - center_offset - axial_max
    correction = torch.minimum(torch.maximum(torch.zeros_like(low), low), high)
    return target + correction[..., None] * free_axis, quat, raw_delta, axial_min, axial_max


def pregrasp_pitch_constants(cache, pitches):
    """GPU equivalent of pitch candidates, effective grasp, and semantic noise."""
    horizontal = cache["horizontal_orientation"][:, None].expand(-1, len(pitches), -1)
    axis_local = torch.zeros_like(cache["center_standoff"])
    axis_local[:, 1] = 1
    joint_axis = _quat_rotate(cache["horizontal_orientation"], axis_local)
    axis = joint_axis[:, None].expand(-1, len(pitches), -1).flatten(0, 1)
    angle = torch.deg2rad(pitches)[None].expand(len(horizontal), -1).flatten()
    negative = _quat_multiply(_axis_angle_quat(axis, -angle), horizontal.flatten(0, 1))
    positive = _quat_multiply(_axis_angle_quat(axis, angle), horizontal.flatten(0, 1))
    shaft = torch.zeros_like(axis)
    shaft[:, 0] = 1
    choose_negative = _quat_rotate(negative, shaft)[:, 2] <= _quat_rotate(positive, shaft)[:, 2]
    nominal_q = _normalize(torch.where(choose_negative[:, None], negative, positive)).reshape_as(horizontal)
    nominal = cache["center_standoff"][:, None].expand(-1, len(pitches), -1).clone()
    nominal[..., 2] = cache["fixed_surface_target_z"][:, None] - _quat_rotate(
        nominal_q, cache["fixed_inner_center_local"][:, None])[..., 2]
    position, quat, raw_delta, axial_min, axial_max = grasp_noise_target(cache, nominal, nominal_q)
    grasp = cache["grasp_center_pos"][:, None] - _quat_rotate(nominal_q, cache["grasp_ee_offset_local"][:, None])
    grasp[..., 2] = nominal[..., 2]
    grasp, grasp_q, _, _, _ = grasp_noise_target(cache, grasp, nominal_q)
    return dict(nominal=nominal, nominal_q=nominal_q, position=position, quat=quat,
                grasp=grasp, grasp_q=grasp_q, raw_delta=raw_delta, axial_min=axial_min, axial_max=axial_max)


def retract_pose_samples(base, radius, rotation_bound, seeds, entries, start, count=64):
    """Deterministic device-native uniform-volume ball and uniform RPY draws.

    This is a counter-based stream, not NumPy PCG64's bit stream. Environment
    seeds, entry indices and candidate indices are independent of sparse batches.
    """
    candidate = torch.arange(start, start + count, device=base.device, dtype=torch.int64)
    lane = torch.arange(8, device=base.device, dtype=torch.int64)
    counter = (seeds[:, None, None] + 0xA2E72AC7 + entries[:, None, None] * 104729
               + candidate[None, :, None] * 1009 + lane[None, None] * 9176)
    # Integer avalanche in 32 bits; mask after each product avoids signed shifts.
    counter = (counter ^ (counter >> 16)) * 0x45D9F3B & 0xFFFFFFFF
    counter = (counter ^ (counter >> 16)) * 0x45D9F3B & 0xFFFFFFFF
    counter = counter ^ (counter >> 16)
    uniform = (counter.to(torch.float64) + .5) / 4294967296.
    normal_radius = torch.sqrt(-2 * torch.log(uniform[..., :2]))
    normal_angle = 2 * math.pi * uniform[..., 2:4]
    direction = torch.stack((normal_radius[..., 0] * normal_angle[..., 0].cos(),
                             normal_radius[..., 0] * normal_angle[..., 0].sin(),
                             normal_radius[..., 1] * normal_angle[..., 1].cos()), -1)
    delta = _normalize(direction) * (radius[:, None] * uniform[..., 4].pow(1 / 3))[..., None]
    rpy = (2 * uniform[..., 5:8] - 1) * rotation_bound[:, None, None]
    return base[:, None].to(torch.float64) + torch.cat((delta, rpy), -1)


class GPUStagePlanningMixin:
    """Shared GPU planners; initialize once, then operate only on tensors."""

    def initialize_gpu_stage_constants(self):
        # All controller/NumPy access is intentionally confined to initialization.
        import numpy as np
        from gr00t.rl.auto_open.geometry import _make_batched_geometry_cache

        device = self.controllers[0].env.device
        cache = {}

        def tensor(values, dtype=torch.float32):
            return torch.as_tensor(np.asarray(values), device=device, dtype=dtype)

        for name in ("grasp_waypoint_world", "grasp_center_pos", "handle_free_axis_world",
                     "arm_retract_pose_local", "arm_retract_position_noise_radius_m",
                     "arm_retract_rotation_noise_deg"):
            cache[name] = tensor([getattr(c, name) for c in self.controllers])
        cache["retract_seeds"] = tensor([c.arm_retract_random_seed for c in self.controllers], torch.int64)
        cache["retract_entries"] = tensor([c.arm_retract_entry_count for c in self.controllers], torch.int64)
        cache["retract_lower"] = tensor([c.ee_steady_model.lower_limits for c in self.controllers], torch.float64)
        cache["retract_upper"] = tensor([c.ee_steady_model.upper_limits for c in self.controllers], torch.float64)
        cache["geometry"] = _make_batched_geometry_cache(self.controllers, device)
        for name in ("panel_push_world", "panel_half_thickness", "panel_push_preload",
                     "panel_standoff_target_clearance", "panel_hinge_to_free_direction",
                     "panel_hinge_to_free_edge", "panel_bottom_z", "panel_top_z",
                     "panel_push_tangent_offset", "panel_push_height_offset",
                     "panel_contact_edge_margin"):
            cache[name] = tensor([getattr(c, name) for c in self.controllers])
        for key, attribute in (("horizontal_orientation", "grasp_orientation_quat"),
                               ("center_standoff", "center_standoff_ee_world"),
                               ("fixed_surface_target_z", "fixed_surface_target_z"),
                               ("fixed_inner_center_local", "fixed_inner_center_local"),
                               ("grasp_ee_offset_local", "grasp_ee_offset_local")):
            cache[key] = tensor([getattr(c, attribute) for c in self.controllers])
        for name in ("depth_axis", "separation_axis"):
            cache[name] = tensor([c.open_template[name] for c in self.controllers])
        cache["jaw_points"] = tensor([np.concatenate((c.open_template["fixed_points"],
            c.open_template["moving_points"]), axis=0) for c in self.controllers])
        cache["outer_half_length"] = tensor([c.handle_half_length for c in self.controllers]) + (
            tensor([c.handle_upper_z for c in self.controllers])
            - tensor([c.handle_lower_z for c in self.controllers])) .mul(.5).clamp_min(1.e-4)
        noise_rows = []

        def pose_noise(controller, label, position_bounds, rotation_bounds):
            noise = controller.target_noise
            key = f"pose:{label}"
            sample = noise._samples.get(key)
            if sample is None:
                sample = dict(
                    semantic_position_delta=noise._signed_vector(f"{key}:position", position_bounds),
                    semantic_rotation_delta=noise._signed_vector(f"{key}:rotation", rotation_bounds))
                noise._samples[key] = sample
            return sample["semantic_position_delta"], sample["semantic_rotation_delta"]

        for controller in self.controllers:
            config = controller.target_noise.config
            noise_rows.append(dict(
                grasp=pose_noise(controller, "grasp_chain", (config.grasp_depth_m, config.grasp_handle_axis_m,
                    config.grasp_closing_m), (config.grasp_roll_rad, 0., 0.)),
                release=pose_noise(controller, "open_and_clear", (0., 0., config.release_lift_m), (0., 0., 0.)),
                retreat=pose_noise(controller, "retreat_arm", (config.release_retreat_normal_m, 0.,
                    config.release_retreat_vertical_m), (0., 0., 0.)),
                panel=pose_noise(controller, "panel_push_chain", (config.panel_depth_m, config.panel_tangent_m,
                    config.panel_vertical_m), (config.panel_roll_rad, config.panel_tilt_horizontal_rad,
                    config.panel_tilt_vertical_rad))))
        cache["grasp_position_noise"] = tensor([row["grasp"][0] for row in noise_rows])
        cache["grasp_rotation_noise"] = tensor([row["grasp"][1] for row in noise_rows])
        cache["pitch_deg"] = tensor([0., 5., 10., 15., 20.])
        cache.update(pregrasp_pitch_constants(cache, cache["pitch_deg"]))
        normal = tensor([c.door_normal for c in self.controllers])
        tangent = torch.stack((-normal[:, 1], normal[:, 0], torch.zeros_like(normal[:, 0])), -1)
        for label in ("release", "retreat"):
            semantic = tensor([row[label][0] for row in noise_rows])
            delta = semantic[:, :1] * normal + semantic[:, 1:2] * tangent
            delta[:, 2] += semantic[:, 2]
            cache[f"{label}_delta"] = delta
        cache["panel_position_noise"] = tensor([row["panel"][0] for row in noise_rows])
        cache["panel_rotation_noise"] = tensor([row["panel"][1] for row in noise_rows])
        cache["selected_pitch"] = torch.zeros(len(self.controllers), device=device, dtype=torch.int64)
        self.gpu_stage_constants = cache
        self.gpu_stage_metadata = []
        if not hasattr(self, "path_diagnostics"):
            self.path_diagnostics = []

    def _stage_path(self, fsm, state, rows, bank_name, label, end, end_q,
                    *, start_q=None, steps=None, failure_code=44, samples=None,
                    inverse_tolerance_m=None):
        bank = getattr(fsm, bank_name)
        if samples is None:
            samples = path_samples(
                state["ee_pos"][rows], end,
                state["ee_orn"][rows] if start_q is None else start_q,
                end_q, bank.capacity, steps=steps)
        positions, orientations, lengths = samples
        valid = (lengths <= bank.capacity) & torch.isfinite(positions).all(dim=(1, 2))
        valid &= torch.isfinite(orientations).all(dim=(1, 2))
        rejected = rows[~valid]
        fsm.fail(rejected, failure_code)
        rows, positions, orientations, lengths = rows[valid], positions[valid], orientations[valid], lengths[valid]
        if not len(rows):
            self._record_stats(label, 0, errors=torch.tensor(len(rejected), device=fsm.device))
            return rows, valid
        result, owner, sample = self._inverse_pose_bank(
            fsm, state, rows, positions, orientations, lengths)
        positions[owner, sample] = result.command_position_world.to(positions.dtype)
        orientations[owner, sample] = result.command_orientation_world_wxyz.to(orientations.dtype)
        finite = torch.isfinite(positions).all(dim=(1, 2)) & torch.isfinite(orientations).all(dim=(1, 2))
        if inverse_tolerance_m is not None:
            error = torch.linalg.vector_norm(result.position_error, dim=-1)
            angle = torch.linalg.vector_norm(result.orientation_error, dim=-1)
            worst_position = torch.zeros(len(rows), device=fsm.device, dtype=error.dtype)
            worst_rotation = torch.zeros_like(worst_position)
            worst_position.scatter_reduce_(0, owner, torch.nan_to_num(error, nan=torch.inf), reduce='amax')
            worst_rotation.scatter_reduce_(0, owner, torch.nan_to_num(angle, nan=torch.inf), reduce='amax')
            finite &= (worst_position <= inverse_tolerance_m) & (worst_rotation <= 1.e-4)
            self.path_diagnostics.append((label + '_precision', rows.clone(), {
                'max_position_error_m': worst_position, 'max_orientation_error': worst_rotation,
                'within_tolerance': finite.clone()}))
        failed = rows[~finite]
        fsm.fail(failed, failure_code)
        valid[valid.clone()] = finite
        rows, positions, orientations, lengths = rows[finite], positions[finite], orientations[finite], lengths[finite]
        bank.positions[rows] = positions.to(bank.positions.dtype)
        bank.orientations[rows] = orientations.to(bank.orientations.dtype)
        bank.length[rows] = lengths.to(bank.length.dtype)
        bank.index[rows] = 0
        self.path_diagnostics.append((label, rows.clone(), {"lengths": lengths.clone()}))
        self._record_stats(label, len(rows), errors=torch.tensor(len(rejected) + len(failed), device=fsm.device))
        return rows, valid

    def plan_pregrasp(self, fsm, device_state, rows):
        if not len(rows):
            return
        cache = self.gpu_stage_constants
        count = cache["position"].shape[1]
        candidate_rows = rows.repeat_interleave(count)
        start = cache["position"][rows].flatten(0, 1)
        start_q = cache["quat"][rows].flatten(0, 1)
        bodies = device_state["door_body_state_w"]
        panel_q = bodies[candidate_rows, fsm.panel_body_idx, 3:7]
        axis = _quat_rotate(panel_q, fsm.handle_joint_axis_panel_local[candidate_rows])
        handle_start = device_state["door_joint_position"][candidate_rows, fsm.handle_joint_idx]
        hinge_angle = fsm.hinge_axis_sign[candidate_rows] * (
            fsm.turn_gap_angle[candidate_rows] + math.radians(2) + fsm.turn_gap_margin[candidate_rows])
        position, orientation, candidate = pregrasp_pose_family(
            start, start_q, cache["grasp"][rows].flatten(0, 1), cache["grasp_q"][rows].flatten(0, 1),
            bodies[candidate_rows, fsm.handle_body_idx, :3], axis,
            fsm.hinge_world[candidate_rows], hinge_angle, handle_start,
            .74 + fsm.turn_target_margin[candidate_rows])
        # Treat packed samples as one-pose paths to reuse the coordinator's inverse.
        result, owner, _ = self._inverse_pose_bank(
            fsm, device_state, candidate_rows[candidate], position[:, None], orientation[:, None],
            torch.ones(len(position), device=fsm.device, dtype=torch.int64))
        position_error = torch.linalg.vector_norm(result.position_error, dim=-1)
        rotation_error = torch.linalg.vector_norm(result.orientation_error, dim=-1)
        bad = (~torch.isfinite(position_error) | ~torch.isfinite(rotation_error)
               | (position_error > 1.e-4) | (rotation_error > 1.e-4))
        failed = torch.zeros(len(candidate_rows), device=fsm.device, dtype=torch.int64)
        failed.scatter_add_(0, candidate[owner], bad.long())
        self.gpu_stage_metadata.append(("pregrasp_errors", rows.clone(), {
            "candidate": candidate[owner].clone(), "position_error": position_error.clone(),
            "orientation_error": rotation_error.clone(), "positions": position[owner].clone(),
            "orientations": orientation[owner].clone(), "pitch_deg": cache["pitch_deg"].clone()}))
        exact = failed.reshape(-1, count) == 0
        accepted = exact.any(-1)
        selected = exact.long().argmax(-1)
        self.path_diagnostics.append(("pregrasp_pitch", rows.clone(), {
            "exact": exact, "selected_pitch_deg": cache["pitch_deg"][selected], "failed_samples": failed.reshape(-1, count)}))
        fsm.fail(rows[~accepted], 47)
        rows, selected = rows[accepted], selected[accepted]
        if not len(rows):
            return
        cache["selected_pitch"][rows] = selected
        end, end_q = cache["position"][rows, selected], cache["quat"][rows, selected]
        points, quats, diagnostics = pregrasp_avoidance_waypoints(
            device_state, cache['geometry'], rows, cache['geometry']['door_normal'][rows], end, end_q)
        positions, orientations, lengths, overflow = self.pull_geometry._sample(
            points, quats, torch.full_like(rows, 4))
        capacity = fsm.arm_align_path.capacity
        # The common sampler reports overflow separately; never accept truncation.
        lengths = torch.where(overflow, torch.full_like(lengths, capacity + 1), lengths)
        self.path_diagnostics.append(('pregrasp_avoidance', rows.clone(), diagnostics))
        planned, valid = self._stage_path(fsm, device_state, rows, "arm_align_path", "arm_align_path",
            end, end_q, failure_code=40,
            samples=(positions[:, :capacity], orientations[:, :capacity], lengths),
            inverse_tolerance_m=1.e-4)
        pitch = selected[valid]
        self.gpu_stage_metadata.append(("pregrasp", planned.clone(), {
            "pitch_deg": cache["pitch_deg"][pitch], "orientation": cache["nominal_q"][planned, pitch],
            "end_position": cache["nominal"][planned, pitch], "desired_target": end[valid]}))
        fsm.arm_align_desired_target_world[planned] = end[valid]
        fsm.arm_align_target_valid[planned] = True
        fsm.arm_stability.clear(planned)
        fsm.transition(planned, fsm.Stage.PREGRASP)
        fsm.stage_steps[planned] = 1

    def plan_center(self, fsm, device_state, rows):
        if not len(rows):
            return
        from gr00t.rl.auto_open.geometry import _batched_geometry_metrics_gpu

        cache = self.gpu_stage_constants
        metrics = _batched_geometry_metrics_gpu(device_state, cache["geometry"], rows, as_numpy=False)
        handle_pose = device_state["door_body_state_w"][rows, fsm.handle_body_idx]
        center = handle_pose[:, :3] + _quat_rotate(handle_pose[:, 3:7], fsm.handle_center_local[rows])
        fixed_upper = metrics["fixed_center"][:, 2] >= center[:, 2]
        gap = torch.where(fixed_upper, metrics["upper_gap"], metrics["lower_gap"])
        shift, feasible = center_vertical_shift(metrics["channel_vertical_error"], gap)
        finite = torch.isfinite(shift) & torch.isfinite(gap) & torch.isfinite(center).all(-1)
        fsm.fail(rows[~finite], 41)
        fsm.fail(rows[finite & ~feasible], 42)
        rows, shift = rows[finite & feasible], shift[finite & feasible]
        pitch = cache["selected_pitch"][rows]
        target = cache["grasp_waypoint_world"][rows].clone()
        target[:, 2] = cache["nominal"][rows, pitch, 2] + shift
        target += cache["raw_delta"][rows, pitch]
        free_axis = cache["handle_free_axis_world"][rows]
        axial = ((target - cache["grasp_center_pos"][rows]) * free_axis).sum(-1)
        low = -cache["outer_half_length"][rows] + .025 - axial - cache["axial_min"][rows, pitch]
        high = cache["outer_half_length"][rows] - .015 - axial - cache["axial_max"][rows, pitch]
        target += torch.minimum(torch.maximum(torch.zeros_like(low), low), high)[:, None] * free_axis
        quat = cache["quat"][rows, pitch]
        nominal_q = cache["nominal_q"][rows, pitch]
        planned, valid = self._stage_path(fsm, device_state, rows, "arm_forward_path", "arm_forward_path",
                                         target, quat, start_q=nominal_q, failure_code=43)
        fsm.arm_forward_desired_target_world[planned] = target[valid]
        bank = fsm.arm_forward_path
        endpoint = bank.length[planned].long() - 1
        fsm.arm_forward_command_world[planned] = bank.positions[planned, endpoint]
        fsm.arm_forward_command_quat[planned] = bank.orientations[planned, endpoint]
        fsm.arm_forward_target_valid[planned] = True
        fsm.arm_stability.clear(planned)
        fsm.handle_align_settle_steps[planned] = 0
        fsm.transition(planned, fsm.Stage.CENTER_HANDLE)
        fsm.stage_steps[planned] = 1

    def plan_release_open(self, fsm, device_state, rows):
        if not len(rows):
            return
        end = device_state["ee_pos"][rows] + self.gpu_stage_constants["release_delta"][rows]
        end[:, 2] += fsm.release_vertical_delta[rows]
        planned, _ = self._stage_path(fsm, device_state, rows, "release_open_path", "release_open_path",
                                     end, device_state["ee_orn"][rows], steps=torch.full_like(rows, 35))
        fsm.release_stable_steps[planned] = 0
        fsm.transition(planned, fsm.Stage.OPEN_AND_CLEAR)
        fsm.stage_steps[planned] = 1

    def plan_retreat(self, fsm, device_state, rows):
        if not len(rows):
            return
        end = (device_state["ee_pos"][rows] + .10 * fsm.door_normal[rows]
               + self.gpu_stage_constants["retreat_delta"][rows])
        end[:, 2] += .02
        steps = torch.ceil(torch.linalg.vector_norm(end - device_state["ee_pos"][rows], dim=-1) / .002).long().clamp_min(50)
        planned, valid = self._stage_path(fsm, device_state, rows, "release_retreat_path", "release_retreat_path",
                                         end, device_state["ee_orn"][rows], steps=steps)
        fsm.release_retreat_target_world[planned] = end[valid]
        fsm.release_stable_steps[planned] = 0
        fsm.transition(planned, fsm.Stage.RETREAT_ARM)
        fsm.stage_steps[planned] = 1

    def _stage_candidate_ik(self, fsm, state, rows, positions, orientations):
        from gr00t.rl.auto_open.planning_ik import solve_candidate_ik
        from gr00t.rl.envs.door.b2z1_reset_geometry import quaternion_xyzw_to_matrix

        cache = self.gpu_stage_constants
        if not len(rows):
            empty = torch.empty(0, device=fsm.device, dtype=torch.float64)
            return empty.bool(), empty, empty, empty
        root = state["root_states"][rows].to(torch.float64)
        lower, upper = cache["retract_lower"][rows], cache["retract_upper"][rows]
        solution, position_error, orientation_error = solve_candidate_ik(
            root[:, :3], root[:, [4, 5, 6, 3]], positions.to(torch.float64),
            quaternion_xyzw_to_matrix(orientations.to(torch.float64)[:, [1, 2, 3, 0]]),
            state["dof_pos"][rows][:, self.arm_joint_ids].to(torch.float64),
            lower, upper, iterations=120, damping=.03)
        margin = torch.minimum(solution - lower, upper - solution).amin(-1)
        reachable = ((position_error <= 1.e-4) & (orientation_error <= 2.e-4) & (margin >= .002))
        return reachable, position_error, orientation_error, margin

    def _panel_handle_bounds(self, fsm, device_state, rows):
        handle_pose = device_state["door_body_state_w"][rows, fsm.handle_body_idx, :7]
        center = handle_pose[:, :3] + _quat_rotate(handle_pose[:, 3:7], fsm.handle_center_local[rows])
        axis = _quat_rotate(handle_pose[:, 3:7], fsm.handle_axis_local[rows])
        half_height = axis[:, 2].abs()*fsm.handle_half_length[rows]+fsm.handle_radius[rows]
        return torch.stack((center[:, 2]-half_height,center[:, 2]+half_height),-1)

    def plan_contact(self, fsm, device_state, rows):
        if not len(rows):
            return
        cache = self.gpu_stage_constants
        geometry = cache["geometry"]
        angle = fsm.hinge_axis_sign[rows] * device_state["door_joint_position"][rows, fsm.hinge_idx]
        rotation = _yaw_quat(angle)
        normal_sign = fsm.panel_contact_normal_sign[rows]
        normal = _quat_rotate(rotation, fsm.door_normal[rows]) * normal_sign[:, None]
        tangent = torch.stack((-normal[:, 1], normal[:, 0], torch.zeros_like(angle)), -1)
        seed = torch.where(fsm.panel_contact_orientation_seed_valid[rows, None],
                           fsm.panel_contact_orientation_seed[rows], device_state["ee_orn"][rows])
        fsm.panel_contact_orientation_seed_valid[rows] = False
        aligned = align_vectors(_quat_rotate(seed, fsm.depth_axis_local[rows]), -normal, seed)
        # Semantic noise is expressed in current normal/tangent/world-up axes.
        noise = cache["panel_position_noise"][rows]
        position_delta = noise[:, :1] * normal + noise[:, 1:2] * tangent
        position_delta[:, 2] += noise[:, 2]
        noise_q = cache["panel_rotation_noise"][rows]
        rotation_vector = noise_q[:, :1] * normal + noise_q[:, 1:2] * tangent
        rotation_vector[:, 2] += noise_q[:, 2]
        rotation_angle = torch.linalg.vector_norm(rotation_vector, dim=-1)
        noise_rotation = _axis_angle_quat(_normalize(rotation_vector), rotation_angle)
        orientation = _quat_multiply(noise_rotation, aligned)
        # Recover current collision points in EE coordinates, then reorient them.
        bodies = device_state["robot_body_state_w"]
        fixed_pose = bodies[rows, geometry["fixed_body_idx"], :7]
        moving_pose = bodies[rows, geometry["moving_body_idx"], :7]
        fixed = fixed_pose[:, None, :3] + _quat_rotate(
            fixed_pose[:, None, 3:7], geometry["fixed_points"][None].expand(len(rows), -1, -1))
        moving = moving_pose[:, None, :3] + _quat_rotate(
            moving_pose[:, None, 3:7], geometry["moving_points"][None].expand(len(rows), -1, -1))
        relative = torch.cat((fixed, moving), 1) - device_state["ee_pos"][rows, None]
        local = _quat_rotate(_quat_conjugate(device_state["ee_orn"][rows])[:, None], relative)
        offsets = _quat_rotate(orientation[:, None], local)
        min_normal = (offsets * normal[:, None]).sum(-1).amin(-1)
        surface = fsm.hinge_world[rows] + _quat_rotate(rotation,
            cache["panel_push_world"][rows] + fsm.door_normal[rows] * cache["panel_half_thickness"][rows, None] * normal_sign[:, None]
            - fsm.hinge_world[rows])
        current_tangent = _quat_rotate(rotation, cache["panel_hinge_to_free_direction"][rows])
        nominal_tangent_offset = cache["panel_push_tangent_offset"][rows]
        if fsm.pull_door:
            forward_offset = fsm.pull_station_config.panel_contact_forward_offset_m
            baseline_mode = fsm.pull_station_config.panel_contact_baseline
            if baseline_mode != "base_forward":
                forward_offset = fsm.pull_base_points[:, 0].amax()
                if baseline_mode == "trunk_front_minus10":
                    forward_offset = forward_offset - .10
            nominal_tangent_offset = panel_forward_anchor_distance(
                device_state["root_states"][rows], fsm.hinge_world[rows], current_tangent,
                forward_offset)
            surface += (nominal_tangent_offset - cache["panel_push_tangent_offset"][rows])[:, None] * current_tangent
        baseline = surface + (cache["panel_standoff_target_clearance"][rows] - min_normal)[:, None] * normal
        baseline += position_delta
        handle_z_bounds = self._panel_handle_bounds(fsm, device_state, rows)
        planned = self._forecast_panel_push(
            fsm, device_state, rows, baseline, orientation, normal,
            current_tangent, local, handle_z_bounds)
        self._publish_panel_contact(fsm, device_state, *planned)

    def _publish_panel_contact(self, fsm, device_state, rows, desired, orientation, normal):
        if not len(rows):
            return
        cache = self.gpu_stage_constants
        if not fsm.pull_door:
            selected = torch.arange(len(rows),device=fsm.device)
            last = (fsm.panel_arrival_path.length[rows]-1).long()
            command_positions = fsm.panel_arrival_path.positions[rows,last]
            command_orientations = fsm.panel_arrival_path.orientations[rows,last]
            errors = 0
        else:
            result, owner, _ = self._inverse_pose_bank(fsm, device_state, rows, desired[:, None], orientation[:, None],
                torch.ones_like(rows))
            exact = ((torch.linalg.vector_norm(result.position_error, dim=-1) <= .010)
                     & (torch.linalg.vector_norm(result.orientation_error, dim=-1) <= 1.e-4))
            fsm.fail(rows[owner[~exact]], 45)
            selected = owner[exact]
            command_positions = result.command_position_world[exact]
            command_orientations = result.command_orientation_world_wxyz[exact]
            errors = (~exact).sum()
        planned = rows[selected]
        fsm.panel_contact_target_world[planned] = desired[selected] - (
            cache["panel_push_preload"][planned] + cache["panel_standoff_target_clearance"][planned])[:, None] * normal[selected]
        fsm.panel_contact_desired_ee_world[planned] = desired[selected]
        fsm.panel_contact_command_target_world[planned] = command_positions.to(torch.float32)
        fsm.panel_contact_command_orientation_world[planned] = command_orientations.to(torch.float32)
        initialize = planned[~fsm.ee_command_valid[planned]]
        fsm.ee_command_world[initialize] = fsm.panel_contact_command_target_world[initialize]
        fsm.ee_orientation_command[initialize] = fsm.panel_contact_command_orientation_world[initialize]
        fsm.ee_command_valid[initialize] = True
        fsm.panel_contact_plan_valid[planned] = True
        self._record_stats("panel_contact", len(planned), errors=errors)

    def _forecast_panel_push(self, fsm, state, rows, desired, orientation, normal,
                             tangent, jaw_local, handle_z_bounds):
        from gr00t.rl.auto_open.panel_motion import fingertip_arc_position
        from gr00t.rl.auto_open.panel_push_forecast import (
            panel_candidate_bank, PANEL_PUSH_JOINT_MARGIN_RAD, select_panel_arc)

        cache = self.gpu_stage_constants
        pull_door = getattr(fsm, 'pull_door', False)
        sign = fsm.hinge_axis_sign[rows].double()
        bank = panel_candidate_bank(
                desired, orientation, normal, tangent, jaw_local, fsm.depth_axis_local[rows],
                fsm.hinge_world[rows],state['root_states'][rows],fsm.env.get_goal_center_offset_local()[rows],
                cache['panel_standoff_target_clearance'][rows],cache['panel_hinge_to_free_edge'][rows],
                cache['panel_bottom_z'][rows],cache['panel_top_z'][rows],cache['panel_contact_edge_margin'][rows],
                fsm.push_comfort_reach_radius[rows],handle_z_bounds,sign,
                height_upper_offset_m=.20 if pull_door else .30)
        targets,quats = bank['targets'],bank['quats']
        tip_offset,tip_surface = bank['tip_offset'],bank['tip_surface']
        candidate_valid,parameters = bank['geometry_valid'],bank['parameters']
        count = targets.shape[1]
        # Fixed-size buckets reuse candidate-IK CUDA graphs across request sizes.
        bucket = 1 << (len(rows)-1).bit_length()
        owner = torch.arange(bucket,device=fsm.device).clamp_max(len(rows)-1)
        reachable, _, _, _ = self._stage_candidate_ik(
            fsm,state,rows[owner].repeat_interleave(count),
            targets[owner].flatten(0,1),quats[owner].flatten(0,1))
        static_valid = reachable.reshape(bucket,count)[:len(rows)]
        start = state['door_joint_position'][rows, fsm.hinge_idx].double()
        angle = torch.minimum(start[:, None] + torch.deg2rad(
            torch.arange(91, device=fsm.device, dtype=torch.float64))[None],
            torch.full_like(start[:, None], math.radians(89.)))
        angle = angle[:, None].expand(-1, count, -1)
        arc = fingertip_arc_position(
            tip_surface[:, :, None], tip_offset[:, :, None],
            fsm.hinge_world[rows].double()[:, None, None],
            sign[:, None, None] * (angle - start[:, None, None]))
        root = state['root_states'][rows].double()
        w, x, y, z = root[:, 3:7].unbind(-1)
        base_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        origin = root[:, :3].clone()
        origin[:, 2] = 0.
        local = _quat_rotate(_yaw_quat(-base_yaw)[:, None, None], arc - origin[:, None, None])
        local -= fsm.env.get_goal_center_offset_local()[rows].double()[:, None, None]
        outside = torch.linalg.vector_norm(local[..., :2], dim=-1) > fsm.push_comfort_reach_radius[rows, None, None]
        index = torch.arange(91, device=fsm.device)[None, None]
        first_exit = torch.where(outside, index, 90).amin(-1)
        first_goal = torch.where(angle >= math.radians(89.) - 1.e-8, index, 90).amin(-1)
        lengths = torch.minimum(first_exit, first_goal) + 2
        lengths = torch.where(candidate_valid & static_valid,lengths,torch.zeros_like(lengths))
        positions = torch.cat((targets[:, :, None], arc), dim=2).flatten(0, 1)
        orientations = quats[:, :, None].expand(-1, -1, positions.shape[1], -1).flatten(0, 1)
        result, owner, sample = self._inverse_pose_bank(
            fsm, state, rows.repeat_interleave(count), positions, orientations, lengths.flatten())
        errors = torch.linalg.vector_norm(result.position_error, dim=-1)
        rotation_error = 2 * torch.asin(torch.linalg.vector_norm(result.orientation_error, dim=-1).clamp(0., 1.))
        limits = self.joint_limits.to(result.arm_joint_positions)
        joint_margin = torch.minimum(result.arm_joint_positions - limits[:, 0],
                                     limits[:, 1] - result.arm_joint_positions).amin(-1)
        good = (torch.isfinite(errors) & (errors <= .001) & (rotation_error <= .01)
                & (joint_margin >= PANEL_PUSH_JOINT_MARGIN_RAD))
        entry_exact = (torch.linalg.vector_norm(result.orientation_error, dim=-1) <= 1.e-4)
        good &= (sample != 0) | entry_exact
        valid = torch.ones(positions.shape[:2], dtype=torch.bool, device=fsm.device)
        valid[owner, sample] = good
        if not pull_door and hasattr(fsm,'panel_arrival_path'):
            from gr00t.rl.auto_open.panel_arrival import validate_panel_arrivals
            bank.update(normal=normal.double(),tangent=tangent.double(),jaw_local=jaw_local.double())
            entry_joints = state['dof_pos'][rows][:,self.arm_joint_ids].double()[:,None].expand(-1,count,-1).clone()
            first = sample==0
            entry_joints.flatten(0,1)[owner[first]] = result.arm_joint_positions[first].to(entry_joints.dtype)
            bank['entry_joints'] = entry_joints
            selected,accepted,stop,complete = validate_panel_arrivals(
                self,fsm,state,rows,bank,valid.reshape(len(rows),count,-1),lengths,angle,
                candidate_valid & static_valid)
        else:
            selected, accepted, stop, complete = select_panel_arc(
                valid.reshape(len(rows), count, -1), lengths, angle, candidate_valid & static_valid,
                minimum_progress_rad=math.radians(5.) if pull_door else 0.,
                prefer_longest=not pull_door)
        ids = torch.arange(len(rows), device=fsm.device)
        diagnostics = {
            'selected_offset': parameters[ids,selected], 'accepted': accepted, 'complete_arc': complete,
            'stop_hinge_angle_rad': stop, 'candidate_geometry_valid': candidate_valid,
            'candidate_static_valid': static_valid,
            'candidate_first_bad_sample': torch.where(
                ~valid.reshape(len(rows), count, -1),
                torch.arange(valid.shape[1], device=fsm.device)[None, None], valid.shape[1]).amin(-1)}
        last_checked = torch.minimum(
            (diagnostics['candidate_first_bad_sample'][ids,selected]-2).clamp(0,90),
            (lengths[ids,selected]-2).clamp(0,90))
        diagnostics['selected_checked_sweep_rad'] = angle[ids,selected,last_checked]-start
        diagnostics['minimum_progress_rad'] = torch.full_like(start,math.radians(5.) if pull_door else 0.)
        diagnostics['workspace_radial_bounds_m'] = bank['radial_bounds']
        self.path_diagnostics.append(('panel_push_forecast', rows.clone(), diagnostics))
        fsm.fail(rows[~accepted], 45)
        fsm.push_planned_stop_hinge[rows[accepted]] = stop[accepted].float()
        fsm.panel_push_tip_offset_local[rows[accepted]] = _quat_rotate(
            _quat_conjugate(quats[ids, selected][accepted]),
            tip_offset[ids, selected][accepted]).float()
        fsm.panel_push_tip_surface_world[rows[accepted]] = tip_surface[ids, selected][accepted].float()
        fsm.panel_push_tip_hinge_angle[rows[accepted]] = start[accepted].float()
        return (rows[accepted], targets[ids, selected][accepted].float(),
                quats[ids, selected][accepted].float(), normal[accepted])

    def _sample_gpu_retract(self, fsm, state, rows):
        from gr00t.rl.auto_open.planning_ik import solve_candidate_ik
        from gr00t.rl.envs.door.b2z1_reset_geometry import quaternion_xyzw_to_matrix

        cache = self.gpu_stage_constants
        entries = cache["retract_entries"][rows].clone()
        cache["retract_entries"][rows] += 1
        accepted = torch.zeros(len(rows), dtype=torch.bool, device=fsm.device)
        poses = torch.zeros((len(rows), 6), device=fsm.device, dtype=torch.float64)
        draws = torch.full_like(rows, 4096)
        evaluated = torch.full_like(rows, 4096)
        accepted_errors = torch.full((len(rows), 3), float("nan"), device=fsm.device, dtype=torch.float64)
        # Fixed candidate chunks, not environment loops; remove accepted rows.
        for start in range(0, 4096, 64):
            active = torch.where(~accepted)[0]
            if not len(active):
                break
            ids = rows[active]
            local = retract_pose_samples(cache["arm_retract_pose_local"][ids],
                cache["arm_retract_position_noise_radius_m"][ids],
                cache["arm_retract_rotation_noise_deg"][ids] * math.pi / 180,
                cache["retract_seeds"][ids], entries[active], start)
            env_ids = ids.repeat_interleave(64)
            reference = fsm.env.get_goal_reference_quat()[env_ids].to(torch.float64)
            origin = fsm.env.get_goal_reference_origin()[env_ids].to(torch.float64)
            offset = fsm.env.get_goal_center_offset_local()[env_ids].to(torch.float64)
            flat = local.flatten(0, 1)
            world = origin + _quat_rotate(reference, flat[:, :3] + offset)
            quat = _quat_multiply(reference, _euler_xyz_quat(flat[:, 3:]))
            root = state["root_states"][env_ids].to(torch.float64)
            lower, upper = cache["retract_lower"][env_ids], cache["retract_upper"][env_ids]
            solution, position_error, orientation_error = solve_candidate_ik(
                root[:, :3], root[:, [4, 5, 6, 3]], world,
                quaternion_xyzw_to_matrix(quat[:, [1, 2, 3, 0]]),
                state["dof_pos"][env_ids][:, self.arm_joint_ids].to(torch.float64),
                lower, upper, iterations=120, damping=.03)
            margin = torch.minimum(solution - lower, upper - solution).amin(-1)
            reachable = ((position_error <= 1.e-4) & (orientation_error <= 2.e-4)
                         & (margin >= .002)).reshape(-1, 64)
            self.gpu_stage_metadata.append(("retract_candidates", ids.clone(), {
                "entry_index": entries[active].clone(), "poses": local,
                "position_error": position_error.reshape(-1, 64),
                "orientation_error": orientation_error.reshape(-1, 64),
                "joint_margin": margin.reshape(-1, 64), "reachable": reachable,
                "candidate_start": torch.full_like(ids, start)}))
            found = reachable.any(-1)
            first = reachable.long().argmax(-1)
            selected = active[found]
            poses[selected] = local[found, first[found]]
            accepted[selected] = True
            draws[selected] = start + first[found] + 1
            evaluated[selected] = start + 64
            accepted_errors[selected] = torch.stack((position_error, orientation_error, margin), -1).reshape(-1, 64, 3)[found, first[found]]
        self.gpu_stage_metadata.append(("retract", rows.clone(), {
            "entry_index": entries, "stage": fsm.stage[rows].clone(), "accepted": accepted,
            "draws_until_acceptance": draws, "evaluated_candidates": evaluated,
            "accepted_pose_local": poses.clone(), "accepted_errors": accepted_errors}))
        self.path_diagnostics.append(("arm_retract_randomization", rows.clone(), {
            "entry_index": entries, "accepted": accepted, "draws_until_acceptance": draws,
            "accepted_pose_local": poses.clone()}))
        fsm.fail(rows[~accepted], 46)
        fsm.active_arm_retract_pose_local[rows[accepted]] = poses[accepted].to(torch.float32)
        return rows[accepted]

    def plan_restore(self, fsm, device_state, rows):
        if not len(rows):
            return
        rows = self._sample_gpu_retract(fsm, device_state, rows)
        if not len(rows):
            return
        local = fsm.active_arm_retract_pose_local[rows]
        reference = fsm.env.get_goal_reference_quat()[rows]
        end = fsm.env.get_goal_reference_origin()[rows] + _quat_rotate(
            reference, local[:, :3] + fsm.env.get_goal_center_offset_local()[rows])
        end_q = _quat_multiply(reference, _euler_xyz_quat(local[:, 3:]))
        steps = torch.ceil(torch.linalg.vector_norm(end - device_state["ee_pos"][rows], dim=-1) / .004).long().clamp_min(50)
        planned, valid = self._stage_path(fsm, device_state, rows, "release_restore_path", "release_restore_path",
                                         end, end_q, steps=steps)
        fsm.release_endpoint_world[planned] = end[valid]
        fsm.release_endpoint_quat[planned] = end_q[valid]
        fsm.release_stable_steps[planned] = 0
        fsm.arm_stability.clear(planned)
        pull = fsm.stage[planned] == int(fsm.Stage.PULL_ARM_RETRACT)
        fsm.transition(planned[pull], fsm.Stage.PULL_ARM_RETRACT)
        fsm.transition(planned[~pull], fsm.Stage.RESTORE_GRIPPER)
        fsm.stage_steps[planned] = 1

    def plan_push_retract(self, fsm, device_state, rows):
        if not len(rows):
            return
        if fsm.pull_door:
            fsm.transition(rows, fsm.Stage.PULL_ARM_RETRACT)
            self.plan_restore(fsm, device_state, rows)
            return
        rows = self._sample_gpu_retract(fsm, device_state, rows)
        fsm.push_arm_detached[rows] = True
        fsm.push_recovery_settle_steps[rows] = 0
        fsm.push_detached_ee_goal_local[rows] = device_state["ee_goal_cart_local"][rows]
        fsm.transition(rows, fsm.Stage.PUSH_ARM_RETRACT)

    def finalize_gpu_stage_diagnostics(self):
        """Offline only: restore legacy controller metadata after the run ends."""
        import numpy as np
        from gr00t.rl.auto_open.planning import _apply_pregrasp_candidate

        rejected = {}
        for label, rows, tensors in self.gpu_stage_metadata:
            values = {key: value.detach().cpu().numpy() for key, value in tensors.items()}
            if label == "pregrasp_errors":
                count = len(values["pitch_deg"])
                for index, env_id in enumerate(rows.detach().cpu().tolist()):
                    diagnostics = []
                    for pitch_index, pitch in enumerate(values["pitch_deg"]):
                        selected = values["candidate"] == index * count + pitch_index
                        position_error = values["position_error"][selected]
                        orientation_error = values["orientation_error"][selected]
                        failed = np.flatnonzero(~np.isfinite(position_error) | ~np.isfinite(orientation_error)
                            | (position_error > 1.e-4) | (orientation_error > 1.e-4))
                        first = int(failed[0]) if len(failed) else None
                        diagnostics.append(dict(pitch_deg=float(pitch), exact=not len(failed),
                            sample_count=len(position_error),
                            max_position_error_m=float(np.nan_to_num(position_error, nan=np.inf).max()),
                            max_orientation_error=float(np.nan_to_num(orientation_error, nan=np.inf).max()),
                            first_failure=None if first is None else dict(sample_index=first,
                                position_error_m=float(position_error[first]), orientation_error=float(orientation_error[first])),
                            position_errors_m=position_error.tolist(), orientation_errors=orientation_error.tolist(),
                            desired_positions_world=values["positions"][selected].tolist(),
                            desired_orientations_world_wxyz=values["orientations"][selected].tolist()))
                    self.controllers[env_id].pregrasp_pitch_diagnostics = diagnostics
                continue
            for index, env_id in enumerate(rows.detach().cpu().tolist()):
                controller = self.controllers[env_id]
                if label == "pregrasp":
                    candidate = {key: values[key][index] for key in ("pitch_deg", "orientation", "end_position")}
                    _apply_pregrasp_candidate(controller, candidate)
                    controller.arm_align_desired_target_world = values["desired_target"][index].copy()
                elif label == "retract_candidates":
                    key = (env_id, int(values["entry_index"][index]))
                    records = rejected.setdefault(key, [])
                    base = np.asarray(controller.arm_retract_pose_local)
                    for sample in np.flatnonzero(~values["reachable"][index]):
                        pose = values["poses"][index, sample]
                        records.append(dict(pose_local=pose.tolist(), position_delta_m=(pose[:3] - base[:3]).tolist(),
                            rpy_delta_rad=(pose[3:] - base[3:]).tolist(),
                            position_error_m=float(values["position_error"][index, sample]),
                            orientation_error_rad=float(values["orientation_error"][index, sample]),
                            joint_margin_rad=float(values["joint_margin"][index, sample])))
                elif label == "retract":
                    entry = int(values["entry_index"][index])
                    accepted = bool(values["accepted"][index])
                    pose = values["accepted_pose_local"][index]
                    errors = values["accepted_errors"][index]
                    base = np.asarray(controller.arm_retract_pose_local)
                    stage = self.Stage(int(values["stage"][index]))
                    # Restore is also entered from RETREAT_ARM; preserve its label.
                    if stage not in (self.Stage.PULL_ARM_RETRACT, self.Stage.PUSH_ARM_RETRACT):
                        stage = self.Stage.RESTORE_GRIPPER
                    rejected_rows = rejected.pop((env_id, entry), [])
                    record = dict(entry_index=entry, stage=stage.name,
                        evaluated_candidates=int(values["evaluated_candidates"][index]),
                        unreachable_candidates=len(rejected_rows), rejected=rejected_rows,
                        rejected_before_acceptance=int(values["draws_until_acceptance"][index]) - 1 if accepted else 4096,
                        draws_until_acceptance=int(values["draws_until_acceptance"][index]) if accepted else None,
                        accepted_pose_local=pose.tolist() if accepted else None,
                        accepted_position_delta_m=(pose[:3] - base[:3]).tolist() if accepted else None,
                        accepted_rpy_delta_rad=(pose[3:] - base[3:]).tolist() if accepted else None,
                        accepted_position_error_m=float(errors[0]) if accepted else None,
                        accepted_orientation_error_rad=float(errors[1]) if accepted else None,
                        accepted_joint_margin_rad=float(errors[2]) if accepted else None)
                    controller.arm_retract_randomization_records.append(record)
                    controller.arm_retract_entry_count = entry + 1
                    if accepted:
                        controller.active_arm_retract_pose_local = pose.astype(np.float32)
        self.gpu_stage_metadata.clear()
