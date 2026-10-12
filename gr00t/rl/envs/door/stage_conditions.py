import torch


def excessive_relative_speed_penalty(
    relative_velocity: torch.Tensor,
    safe_speed: float = 0.2,
) -> torch.Tensor:
    """Return a smooth normalized penalty above a safe relative speed."""
    if relative_velocity.shape[-1] != 3:
        raise ValueError("relative_velocity must contain three Cartesian components")
    if safe_speed <= 0.0:
        raise ValueError("safe_speed must be positive")
    speed = torch.linalg.vector_norm(relative_velocity, dim=-1)
    normalized_excess = ((speed - safe_speed) / safe_speed).clamp(min=0.0)
    return normalized_excess.square()


def walk_to_door_completion_condition(
    operation_distance: torch.Tensor,
    base_yaw_error: torch.Tensor,
    handle_contact_free: torch.Tensor,
    panel_contact_free: torch.Tensor,
    base_motion: torch.Tensor,
    middle_arm_error: torch.Tensor,
    middle_arm_speed: torch.Tensor,
) -> torch.Tensor:
    """Accept a stable operation station with the arm at its middle preset."""
    return (
        (operation_distance < 0.16)
        & (base_yaw_error < 0.08)
        & handle_contact_free
        & panel_contact_free
        & (base_motion < 0.05)
        & (middle_arm_error < 0.05)
        & (middle_arm_speed < 0.08)
    )


def grasp_completion_condition(
    operation_distance: torch.Tensor,
    base_yaw_error: torch.Tensor,
    precise_bilateral_grasp: torch.Tensor,
) -> torch.Tensor:
    """Accept a physical bilateral grasp while the base remains in range.

    Base motion is intentionally not an input: whole-body adjustment remains
    available throughout grasping.
    """
    return (
        (operation_distance < 0.16)
        & (base_yaw_error < 0.08)
        & precise_bilateral_grasp
    )


def bottleneck_balanced_progress(
    component_progress: torch.Tensor,
    bottleneck_weight: float = 0.5,
) -> torch.Tensor:
    """Give every result component credit while emphasizing the weakest one."""
    if component_progress.shape[-1] == 0:
        raise ValueError("component_progress must contain at least one component")
    if not 0.0 <= bottleneck_weight <= 1.0:
        raise ValueError("bottleneck_weight must be in [0, 1]")
    bounded = component_progress.clamp(0.0, 1.0)
    mean_progress = bounded.mean(dim=-1)
    weakest_progress = torch.amin(bounded, dim=-1)
    return (
        (1.0 - bottleneck_weight) * mean_progress
        + bottleneck_weight * weakest_progress
    )


def grasp_collision_aware_position_progress(
    pocket_error_frame: torch.Tensor,
    jaw_closing_coordinates: torch.Tensor,
    handle_radius: torch.Tensor,
    *,
    safe_standoff: float = 0.10,
    clearance_scale: float = 0.03,
    position_scale: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Guide the open gripper around the handle instead of through it.

    A jaw pair that does not yet enclose the handle is attracted to the
    collision-safe normal standoff used by the geometry controller.  As the
    handle becomes enclosed and transversely aligned, that unique normal
    route accumulates credit for reaching the standoff, transverse alignment,
    enclosure, and final approach.  Completed route credit is retained while
    advancing, so improving enclosure at the standoff cannot reduce reward.
    """
    if pocket_error_frame.shape[-1] != 3:
        raise ValueError("pocket_error_frame must contain normal/tangent/vertical")
    if jaw_closing_coordinates.shape[-1] != 2:
        raise ValueError("jaw_closing_coordinates must contain exactly two jaws")
    if safe_standoff <= 0.0 or clearance_scale <= 0.0 or position_scale <= 0.0:
        raise ValueError("grasp navigation scales must be positive")

    jaw_coordinates = torch.sort(jaw_closing_coordinates, dim=-1).values
    safe_radius = handle_radius.clamp(min=1.0e-6)
    jaw_half_span = 0.5 * (jaw_coordinates[..., 1] - jaw_coordinates[..., 0])
    target_frame_jaws = torch.stack(
        [
            pocket_error_frame[..., 2] - jaw_half_span,
            pocket_error_frame[..., 2] + jaw_half_span,
        ],
        dim=-1,
    )
    lower_coverage = (-target_frame_jaws[..., 0] / safe_radius).clamp(0.0, 1.0)
    upper_coverage = (target_frame_jaws[..., 1] / safe_radius).clamp(0.0, 1.0)
    target_frame_between_quality = torch.minimum(lower_coverage, upper_coverage)

    # Door-task +normal points through the doorway.  Collision-safe retreat is
    # toward the robot side, so its positive clearance is the negated error.
    outward_clearance = -pocket_error_frame[..., 0]
    transverse_error = torch.linalg.vector_norm(
        pocket_error_frame[..., 1:], dim=-1
    )
    position_error = torch.linalg.vector_norm(pocket_error_frame, dim=-1)
    transverse_quality = torch.exp(
        -(transverse_error - 0.020).clamp(min=0.0) / position_scale
    )
    direct_position_quality = torch.exp(
        -(position_error - 0.020).clamp(min=0.0) / position_scale
    )
    # Route readiness belongs to the fixed task frame.  Using the handle
    # projection on the current jaw axis here made the normal target change
    # whenever the wrist rotated, even though the EE position was unchanged.
    # Physical enclosure is still evaluated separately by
    # grasp_enclosure_progress using the real current jaw coordinates.
    route_completion = target_frame_between_quality * transverse_quality
    clearance_quality = torch.exp(
        -torch.abs(outward_clearance - safe_standoff) / clearance_scale
    )
    retained_clearance_credit = route_completion + (
        1.0 - route_completion
    ) * clearance_quality
    readiness_progress = 0.5 * (
        transverse_quality + target_frame_between_quality
    )
    retained_readiness_credit = route_completion + (
        1.0 - route_completion
    ) * clearance_quality * readiness_progress
    final_approach_progress = (
        (safe_standoff - outward_clearance) / safe_standoff
    ).clamp(0.0, 1.0)
    progress = (
        retained_clearance_credit
        + retained_readiness_credit
        + route_completion * final_approach_progress
    ) / 3.0
    return (
        progress,
        clearance_quality,
        transverse_quality,
        direct_position_quality,
        target_frame_between_quality,
    )


def grasp_enclosure_progress(
    jaw_closing_coordinates: torch.Tensor,
    handle_radius: torch.Tensor,
    alignment_quality: torch.Tensor,
    physical_contact_quality: torch.Tensor,
    closure_scale: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shape a continuous open, enclose, and close sequence around the handle.

    The geometry is expressed along the gripper closing axis and uses the
    randomized handle radius.  It therefore rewards enough aperture to admit
    the handle without prescribing a gripper angle.  Once the handle lies
    between both jaws, closing toward its two surfaces and valid physical
    contact provide the remaining progress.
    """
    if jaw_closing_coordinates.shape[-1] != 2:
        raise ValueError("jaw_closing_coordinates must contain exactly two jaws")
    if closure_scale <= 0.0:
        raise ValueError("closure_scale must be positive")

    jaw_coordinates = torch.sort(jaw_closing_coordinates, dim=-1).values
    lower_jaw = jaw_coordinates[:, 0]
    upper_jaw = jaw_coordinates[:, 1]
    safe_radius = handle_radius.clamp(min=1.0e-6)

    aperture_quality = (
        (upper_jaw - lower_jaw) / (2.0 * safe_radius)
    ).clamp(0.0, 1.0)
    lower_coverage = (-lower_jaw / safe_radius).clamp(0.0, 1.0)
    upper_coverage = (upper_jaw / safe_radius).clamp(0.0, 1.0)
    between_jaws_quality = torch.minimum(lower_coverage, upper_coverage)

    desired_coordinates = torch.stack([-safe_radius, safe_radius], dim=-1)
    closure_quality = torch.exp(
        -torch.abs(jaw_coordinates - desired_coordinates).mean(dim=-1)
        / closure_scale
    )
    alignment_quality = alignment_quality.clamp(0.0, 1.0)
    physical_contact_quality = physical_contact_quality.clamp(0.0, 1.0)
    aligned_open_quality = alignment_quality * between_jaws_quality
    # Geometry and physical contact are complementary. Using max() here made
    # no-contact, one-jaw, and two-jaw states indistinguishable whenever the
    # geometric closure score was already larger than the sparse contact score.
    capture_completion = 0.5 * closure_quality + 0.5 * physical_contact_quality
    captured_quality = (
        alignment_quality * between_jaws_quality * capture_completion
    )
    progress = (
        0.25 * aperture_quality
        + 0.35 * aligned_open_quality
        + 0.40 * captured_quality
    )
    return (
        progress,
        aperture_quality,
        between_jaws_quality,
        closure_quality,
        captured_quality,
    )


def bilateral_contact_force_progress(
    contact_forces: torch.Tensor,
    valid_contact_geometry: torch.Tensor,
    minimum_force: float = 5.0,
    comfortable_force: float = 30.0,
    excess_force_scale: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Continuously reward safe contact on both jaws and their balance."""
    if contact_forces.shape[-1] != 2:
        raise ValueError("contact_forces must contain exactly two jaws")
    if valid_contact_geometry.shape != contact_forces.shape:
        raise ValueError("valid_contact_geometry must match contact_forces")
    if minimum_force <= 0.0 or excess_force_scale <= 0.0:
        raise ValueError("force scales must be positive")
    if comfortable_force < minimum_force:
        raise ValueError("comfortable_force must not be below minimum_force")

    force = contact_forces.clamp(min=0.0)
    engagement = (force / minimum_force).clamp(max=1.0)
    overload_quality = torch.exp(
        -(force - comfortable_force).clamp(min=0.0) / excess_force_scale
    )
    per_jaw_quality = (
        engagement * overload_quality * valid_contact_geometry.float()
    )
    progress = 0.5 * per_jaw_quality.mean(dim=-1) + 0.5 * torch.amin(
        per_jaw_quality, dim=-1
    )
    return progress, per_jaw_quality


def opposed_handle_contact(
    recent: torch.Tensor,
    stored_points_finite: torch.Tensor,
    stored_axial: torch.Tensor,
    usable_half_length: torch.Tensor,
    stored_closing: torch.Tensor,
    handle_radius: torch.Tensor,
) -> torch.Tensor:
    """Check that both jaws contact valid opposite sides of the handle.

    ContactSensor reports one representative point per collision pair. That
    point may switch within the contact manifold, so the two jaws need not
    report the same axial cross-section while both contacts remain valid.
    """
    return (
        torch.all(recent, dim=-1)
        & stored_points_finite
        & torch.all(
            torch.abs(stored_axial) <= usable_half_length[:, None], dim=-1
        )
        & (stored_closing[:, 0] * stored_closing[:, 1] < 0.0)
        & torch.all(
            torch.abs(stored_closing) >= 0.25 * handle_radius[:, None], dim=-1
        )
    )


def open_latch_completion_condition(
    operation_distance: torch.Tensor,
    base_yaw_error: torch.Tensor,
    latch_position: torch.Tensor,
    hinge_angle: torch.Tensor,
    gap_angle: torch.Tensor,
    precise_bilateral_grasp: torch.Tensor,
) -> torch.Tensor:
    """Require base range, retained grasp, released latch, and a real gap."""
    return (
        (operation_distance < 0.16)
        & (base_yaw_error < 0.08)
        & (latch_position >= 0.02)
        & (hinge_angle >= gap_angle)
        & precise_bilateral_grasp
    )
