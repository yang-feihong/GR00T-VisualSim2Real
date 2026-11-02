"""Replay a recorded grasp plan entirely offline, without starting a simulator."""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from gr00t.rl.auto_open.arm_model import read_arm_joint_limits
from gr00t.rl.auto_open.motion_math import _euler_xyz_quat, _quat_multiply
from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.pick.collision import ArmCollisionSamples
from gr00t.rl.pick.grasp_quality import load_grasp_support
from gr00t.rl.pick.planning import (
    BALANCED_GRASP_PITCH_OFFSETS_DEG,
    inverse_bank,
    plan_carry_path,
    plan_grasp,
)
from gr00t.rl.pick.recorded_states import load_recorded_states
from gr00t.rl.pick.scene_geometry import SceneGeometry, read_scene_triangles
from gr00t.rl.utils.b2z1_asset import ROBOT_URDF


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--collision-model", type=Path)
    parser.add_argument("--frame", type=int, default=-1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--realized-body", action="store_true")
    parser.add_argument("--prefer-realized-body", action="store_true")
    parser.add_argument("--realized-body-tracking-weight", type=float, default=0.0)
    parser.add_argument("--exclude-failed-attempts", action="store_true")
    parser.add_argument("--wrist-band-deg", type=float)
    parser.add_argument("--select-station", action="store_true")
    parser.add_argument("--carry", action="store_true")
    parser.add_argument("--carry-tilt-angles-deg", type=float, nargs="+")
    parser.add_argument("--carry-from-grasp-report", type=Path)
    parser.add_argument("--grasp-index", type=int)
    parser.add_argument("--recorded-grasp-attempt", type=int)
    parser.add_argument("--nudge", action="store_true")
    parser.add_argument("--basket-access", action="store_true")
    parser.add_argument("--closed-nudge-only", action="store_true")
    parser.add_argument("--nudge-contact-height-fractions", type=float, nargs="+")
    parser.add_argument("--recover-grasp-path", action="store_true")
    parser.add_argument("--nudge-shift-samples", type=int, default=24)
    parser.add_argument("--require-sliding-stability", action="store_true")
    parser.add_argument("--basket-local-xy", type=float, nargs=2)
    parser.add_argument("--basket-max-tool-tilt-deg", type=float, default=30.0)
    parser.add_argument("--object-local-rpy-deg", type=float, nargs=3)
    parser.add_argument("--station-proposal", type=Path)
    parser.add_argument("--station-at-recorded-base", action="store_true")
    parser.add_argument("--response-model", type=Path)
    parser.add_argument("--station-pitch-step-deg", type=float, default=15.0)
    parser.add_argument("--station-body-volume", action="store_true")
    parser.add_argument(
        "--station-prefer-standing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--station-require-all-postures", action="store_true")
    parser.add_argument("--refine-orientations", action="store_true")
    parser.add_argument("--retain-depth-candidates", action="store_true")
    parser.add_argument(
        "--natural-orientations", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--object-frame-orientations", action="store_true")
    parser.add_argument("--minimum-pregrasp-standoff-m", type=float, default=0.08)
    parser.add_argument("--capture-axis-pitch-deg", type=float)
    parser.add_argument("--audit-target-path", action="store_true")
    parser.add_argument("--basket-final-entry", action="store_true")
    parser.add_argument("--basket-entry-attempt", type=int, default=0)
    parser.add_argument("--basket-entry-follow-object-from-frame", type=int)
    parser.add_argument("--audit-basket-colliders", action="store_true")
    parser.add_argument(
        "--basket-obstacle-component",
        choices=("all", "basket", "floor", "nonfloor", "scene", "distractors"),
        default="all",
        help="Diagnostic only: isolate jaw clearance obstacles, not collision certification",
    )
    parser.add_argument("--station-body-heights", type=float, nargs="+")
    parser.add_argument("--station-body-pitches-deg", type=float, nargs="+")
    parser.add_argument("--station-yaw-offsets-deg", type=float, nargs="+")
    parser.add_argument("--station-forward-offsets-m", type=float, nargs="+")
    parser.add_argument("--station-lateral-offsets-m", type=float, nargs="+")
    parser.add_argument("--station-ring-radii-m", type=float, nargs="+")
    parser.add_argument("--station-ring-samples", type=int, default=24)
    parser.add_argument(
        "--balanced-pitch-offsets", type=float, nargs="+", default=BALANCED_GRASP_PITCH_OFFSETS_DEG
    )
    parser.add_argument(
        "--surface-distances-only",
        action="store_true",
        help="Diagnostic only: omit TCP-to-surface rays; not collision certification",
    )
    args = parser.parse_args()
    if (
        not math.isfinite(args.basket_max_tool_tilt_deg)
        or not 0 <= args.basket_max_tool_tilt_deg <= 90
    ):
        parser.error("basket-max-tool-tilt-deg must be in [0, 90]")
    if args.carry_from_grasp_report is not None and (not args.carry or args.grasp_index is None):
        parser.error("carry-from-grasp-report requires --carry and --grasp-index")
    if args.closed_nudge_only and not args.nudge:
        parser.error("--closed-nudge-only requires --nudge")
    if args.wrist_band_deg is not None and (
        not math.isfinite(args.wrist_band_deg) or args.wrist_band_deg <= 0
    ):
        parser.error("--wrist-band-deg must be finite and positive")
    if not 0 < args.station_pitch_step_deg <= 30:
        parser.error("station pitch step must be in (0, 30]")
    for values in (args.station_body_heights, args.station_body_pitches_deg):
        if values is not None and not all(math.isfinite(value) for value in values):
            parser.error("body hypotheses must be finite")
    if args.station_body_heights is not None and min(args.station_body_heights) <= 0:
        parser.error("body heights must be positive")
    episode = json.loads((args.run / "episode.json").read_text())
    if args.basket_local_xy is not None and (
        episode.get("task_type") != "pick_basket"
        or not all(math.isfinite(v) for v in args.basket_local_xy)
    ):
        parser.error("basket-local-xy requires a basket episode and finite coordinates")
    trace = load_recorded_states(args.run)
    state = trace[args.frame]
    state = {key: value.cuda() for key, value in state.items()}
    initial_object_height = trace[0]["object_state"][:, 2].cuda()
    if args.object_local_rpy_deg is not None:
        if not all(math.isfinite(value) for value in args.object_local_rpy_deg):
            parser.error("object-local-rpy-deg must be finite")
        root = state["root_states"]
        points = load_grasp_support(episode["asset"]["usd_path"], root.device)
        normal = root.new_tensor([[0, 0, 1]])
        if episode.get("task_type") == "pick_basket":
            normal = _quat_rotate(root[:, 3:7], normal)
        obj = state["object_state"].clone()
        old_minimum = (
            (_quat_rotate(obj[:, None, 3:7], points[None]) * normal[:, None]).sum(-1).amin(-1)
        )
        rotation = _euler_xyz_quat(root.new_tensor([args.object_local_rpy_deg]) * (math.pi / 180))
        obj[:, 3:7] = _quat_multiply(obj[:, 3:7], rotation)
        new_minimum = (
            (_quat_rotate(obj[:, None, 3:7], points[None]) * normal[:, None]).sum(-1).amin(-1)
        )
        shift = (old_minimum - new_minimum)[:, None] * normal
        obj[:, :3] += shift
        initial_object_height = initial_object_height + shift[:, 2]
        state["object_state"] = obj
    if args.basket_local_xy is not None:
        root = state["root_states"]
        inverse = root[:, 3:7].clone()
        inverse[:, 1:] *= -1
        local = _quat_rotate(inverse, state["object_state"][:, :3] - root[:, :3])
        local[:, :2] = local.new_tensor(args.basket_local_xy)
        state["object_state"] = state["object_state"].clone()
        state["object_state"][:, :3] = root[:, :3] + _quat_rotate(root[:, 3:7], local)
    model = torch.load(args.collision_model or args.run / "collision_model.pt", weights_only=False)
    collision = ArmCollisionSamples.__new__(ArmCollisionSamples)
    collision.parts = [(index, points.cuda()) for index, points in model["parts"]]
    collision.arm_joint_ids = model["arm_joint_ids"].cuda()
    if "gripper_surface" in model:
        collision.gripper_surface = [p.cuda() for p in model["gripper_surface"]]
    points, faces = read_scene_triangles(
        Path(episode["scene_usd_path"]), Path(episode["geometry_cache_root"])
    )
    geometry = SceneGeometry(points, faces)
    if "batch_scene_origin_m" in episode:
        from gr00t.rl.pick.geometry_views import TranslatedGeometry

        geometry = TranslatedGeometry(
            geometry, state["ee_pos"].new_tensor(episode["batch_scene_origin_m"])
        )
    basket_controller = None
    if episode.get("task_type") == "pick_basket":
        from gr00t.rl.pick.basket import BasketGeometry
        from gr00t.rl.pick.basket_controller import (
            BasketArmCollisionSamples,
            BasketPickController,
            basket_planning_geometry,
        )

        basket_controller = BasketPickController.__new__(BasketPickController)
        basket_controller.device = limits_device = state["root_states"].device
        basket_controller.episode = episode
        basket_controller.final_entry = False
        basket_controller.awaiting_entry_refresh = False
        basket_controller.entry_orientation = None
        basket_controller.maximum_tool_tilt_deg = args.basket_max_tool_tilt_deg
        basket_controller.basket = BasketGeometry.from_robot_usd(device=limits_device)
        basket_controller.lift_margin = float(episode.get("lift_above_rim_m", 0.10))
        if args.carry_tilt_angles_deg is not None:
            if not args.carry or not all(
                math.isfinite(v) and 0 < v <= 90 for v in args.carry_tilt_angles_deg
            ):
                parser.error("Carry tilt angles require carry mode and angles in (0, 90]")
            basket_controller.carry_tilt_angles_deg = args.carry_tilt_angles_deg
        basket_controller.object_support_points = load_grasp_support(
            episode["asset"]["usd_path"], limits_device
        )
        geometry = basket_planning_geometry(geometry, basket_controller.basket, episode, state)
        if args.basket_obstacle_component != "all":
            from gr00t.rl.pick.distractors import build_distractor_mesh

            if args.basket_obstacle_component in ("basket", "floor", "nonfloor"):
                points, faces = basket_controller.basket.world_mesh(state["root_states"][0])
                if args.basket_obstacle_component != "basket":
                    floor = basket_controller.basket.names.index("back_box_062")
                    floor_faces = faces[:, 0] // 8 == floor
                    faces = faces[
                        floor_faces if args.basket_obstacle_component == "floor" else ~floor_faces
                    ]
                isolated = SceneGeometry(points, faces)
            elif args.basket_obstacle_component == "scene":
                points, faces = read_scene_triangles(
                    Path(episode["scene_usd_path"]), Path(episode["geometry_cache_root"])
                )
                isolated = SceneGeometry(points, faces)
            else:
                mesh = build_distractor_mesh(
                    episode.get("distractors", []), state["distractor_states"]
                )
                isolated = SceneGeometry(mesh["points"], mesh["faces"])
            isolated.nonrobot_geometry = geometry.nonrobot_geometry
            isolated.basket_model = basket_controller.basket
            geometry = isolated
        collision = BasketArmCollisionSamples(collision)
    obstacle_path = args.run / "planning_obstacles.pt"
    if obstacle_path.exists() and basket_controller is None:
        from gr00t.rl.pick.distractors import augment_scene_geometry

        obstacles = torch.load(obstacle_path, weights_only=False, map_location="cuda:0")
        geometry = augment_scene_geometry(geometry, obstacles)
    elif basket_controller is None and episode.get("distractors"):
        from gr00t.rl.pick.distractors import augment_scene_geometry, build_distractor_mesh

        geometry = augment_scene_geometry(
            geometry, build_distractor_mesh(episode["distractors"], state["distractor_states"])
        )
    if args.surface_distances_only:

        def surface_distances(positions, orientations, local, **kwargs):
            world = _quat_rotate(orientations[..., None, :], local) + positions[..., None, :]
            return geometry.distances(world).amin(-1)

        geometry.rigid_surface_clearance = surface_distances
    limits = torch.tensor(
        read_arm_joint_limits(ROBOT_URDF).copy(), device="cuda:0", dtype=torch.float32
    )
    if args.basket_access:
        if basket_controller is None or args.nudge or args.carry or args.select_station:
            parser.error("Basket access requires a basket run without nudge/carry/station mode")
        from gr00t.rl.pick.basket_initialization import basket_grasp_access

        accepted, report = basket_grasp_access(
            state,
            episode,
            basket_controller.basket,
            collision,
            geometry,
            basket_controller.object_support_points,
            limits,
            0.02,
        )
        report.update(accepted=accepted, source_frame=args.frame, source_run=str(args.run))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
        return
    if args.nudge:
        if basket_controller is None or args.carry or args.select_station:
            parser.error("Nudge replay requires a basket run without carry/station selection")
        from gr00t.rl.pick.basket_nudge import plan_clear_for_nudge, plan_closed_nudge

        if args.closed_nudge_only:
            clear = (state["ee_pos"][:, None], state["ee_orn"][:, None])
            clear_report = {"skipped": True, "reason": "use_recorded_closed_jaw_entry"}
        else:
            clear, clear_report = plan_clear_for_nudge(
                state, basket_controller.basket, limits, collision, geometry, 0.02, episode=episode
            )
        nudge = None
        report = {
            "clear": clear_report,
            "valid": False,
            "source_frame": args.frame,
            "source_run": str(args.run),
        }
        if clear is not None:
            virtual = {**state, "ee_pos": clear[0][:, -1], "ee_orn": clear[1][:, -1]}
            nudge, detail = plan_closed_nudge(
                virtual,
                episode,
                basket_controller.basket,
                limits,
                collision,
                geometry,
                0.02,
                current_path_unavailable=args.recover_grasp_path,
                shift_samples=args.nudge_shift_samples,
                require_sliding_stability=args.require_sliding_stability,
                **(
                    {"contact_height_fractions": args.nudge_contact_height_fractions}
                    if args.nudge_contact_height_fractions is not None
                    else {}
                ),
            )
            report.update(
                nudge=detail,
                valid=nudge is not None,
                assumption=(
                    "Use recorded closed-jaw entry directly"
                    if args.closed_nudge_only
                    else "EE starts at the planned clear endpoint; "
                    "actual entry is replanned after clearing"
                ),
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        torch.save({"clear": clear, "nudge": nudge}, args.output.with_suffix(".pt"))
        print(f"[pick:nudge] valid={report['valid']} report={args.output}")
        return
    if args.carry:
        if args.select_station or args.surface_distances_only:
            parser.error("Carry replay requires unmodified geometry and no station selection")
        counterfactual = None
        if args.carry_from_grasp_report is not None:
            if basket_controller is None:
                parser.error("Counterfactual seated grasp requires a basket episode")
            proposal = json.loads(args.carry_from_grasp_report.read_text())
            orientation = limits.new_tensor(proposal["candidate_orientations_world_wxyz"])[
                :, args.grasp_index
            ]
            quality = proposal["convex_contact_model"]
            center = limits.new_tensor(quality["command_center_local"])[:, args.grasp_index]
            position = state["object_state"][:, :3] - _quat_rotate(orientation, center)
            actual = state["dof_pos"][:, collision.arm_joint_ids]
            seeds = torch.cat((actual, limits.mean(-1)[None], limits.sum(-1)[None] - actual))
            solved = inverse_bank(
                position[None].expand(len(seeds), -1, -1),
                orientation[None].expand(len(seeds), -1, -1),
                state["root_states"].expand(len(seeds), -1),
                seeds,
                limits,
            )
            if not bool(solved.exact.any()):
                parser.error("Proposed grasp has no exact solution at the recorded body pose")
            cost = (solved.arm_joint_positions - actual).norm(dim=-1)
            selected = cost.masked_fill(~solved.exact, torch.inf).argmin()
            angle = limits.new_tensor(quality["aperture_angle_rad"])[0, args.grasp_index]
            state = {**state, "ee_pos": position, "ee_orn": orientation}
            state["dof_pos"] = state["dof_pos"].clone()
            state["dof_pos"][:, collision.arm_joint_ids] = solved.arm_joint_positions[selected]
            state["dof_pos"][:, model["gripper_joint_ids"].cuda()] = -angle
            counterfactual = {
                "grasp_report": str(args.carry_from_grasp_report),
                "grasp_index": args.grasp_index,
                "assumption": "Exact virtual seated grasp; not an executed or retained grasp",
            }
        aperture = -state["dof_pos"][:, int(model["gripper_joint_ids"][0])]
        carry_planner = (
            plan_carry_path if basket_controller is None else basket_controller.plan_basket_carry
        )
        paths, diagnostics = carry_planner(
            state,
            torch.zeros_like(state["ee_pos"]),
            initial_object_height,
            aperture,
            limits,
            collision,
            geometry,
            0.02,
        )
        diagnostics.update(
            source_run=str(args.run),
            source_frame=args.frame,
            valid=paths is not None,
            counterfactual_seated_grasp=counterfactual,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(diagnostics, indent=2) + "\n")
        print(json.dumps(diagnostics))
        return
    if args.select_station:
        from gr00t.rl.pick.gaze import sample_approach_offsets
        from gr00t.rl.pick.station import select_initial_pick_station

        destination = limits.new_tensor([[*episode["station_xy"], episode["station_yaw"]]])
        destination[:, 2] += sample_approach_offsets(episode["seed"], limits.device)[0]
        if args.station_at_recorded_base:
            root = state["root_states"]
            w, x, y, z = root[:, 3:7].unbind(-1)
            destination = torch.cat(
                (root[:, :2], torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))[:, None]),
                -1,
            )
        torch.cuda.synchronize()
        started = time.perf_counter()
        station_limits = limits
        if args.wrist_band_deg is not None:
            station_limits = limits.clone()
            actual_q6 = state["dof_pos"][0, collision.arm_joint_ids[-1]]
            band = math.radians(args.wrist_band_deg)
            station_limits[5, 0] = torch.maximum(limits[5, 0], actual_q6 - band)
            station_limits[5, 1] = torch.minimum(limits[5, 1], actual_q6 + band)
        station_options = {
            key: value
            for key, value in (
                ("body_heights", args.station_body_heights),
                ("body_pitches_deg", args.station_body_pitches_deg),
                ("yaw_offsets_deg", args.station_yaw_offsets_deg),
                ("forward_offsets_m", args.station_forward_offsets_m),
                ("lateral_offsets_m", args.station_lateral_offsets_m),
                ("ring_radii_m", args.station_ring_radii_m),
                ("ring_samples", args.station_ring_samples),
            )
            if value is not None
        }
        if args.response_model is not None:
            from gr00t.rl.pick.reach_calibration import CalibratedBodyResponse

            station_options["response_model"] = CalibratedBodyResponse(
                args.response_model, limits.device, allow_unverified=True
            )
        pose, orientation, _, diagnostics = select_initial_pick_station(
            state,
            destination,
            episode["asset"].get("intrinsic_size_m", episode["asset"]["size_m"]),
            load_grasp_support(episode["asset"]["usd_path"], "cuda:0"),
            station_limits,
            collision,
            geometry,
            pitch_step_deg=args.station_pitch_step_deg,
            body_volume=args.station_body_volume,
            prefer_standing=args.station_prefer_standing,
            require_all_postures=args.station_require_all_postures,
            refine_orientations=args.refine_orientations,
            natural_orientations=args.natural_orientations,
            retain_depth_candidates=args.retain_depth_candidates,
            **station_options,
        )
        torch.cuda.synchronize()
        diagnostics.update(
            planning_s=time.perf_counter() - started,
            source_run=str(args.run),
            source_frame=args.frame,
            selected_station=pose,
            selected_grasp_orientation=orientation,
            wrist_band_deg=args.wrist_band_deg,
            wrist_band_reference="recorded_frame_joint_position",
            note="Initial-station endpoint audit only; not a physical grasp result.",
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(diagnostics, indent=2, default=lambda value: value.detach().cpu().tolist())
            + "\n"
        )
        print(f"[pick:station] {args.output}")
        return
    summary_path = args.run / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    initial_station = (
        json.loads(args.station_proposal.read_text())
        if args.station_proposal
        else summary.get("planning", {}).get("initial_station", {})
    )
    preferred = initial_station.get("selected_orientation")
    if args.basket_final_entry:
        if basket_controller is None or args.basket_entry_attempt < 0:
            parser.error("Final-entry replay requires a basket run and nonnegative attempt index")
        recorded = summary["planning"]["grasp_attempts"][args.basket_entry_attempt]
        if recorded.get("selected") is None:
            parser.error("The recorded entry attempt did not select an orientation")
        basket_controller.final_entry = True
        basket_controller.entry_orientation = limits.new_tensor(
            recorded["candidate_orientations_world_wxyz"][0][recorded["selected"]]
        )[None]
        if args.basket_entry_follow_object_from_frame is not None:
            source = (
                trace[args.basket_entry_follow_object_from_frame]["object_state"][:, 3:7]
                .cuda()
                .clone()
            )
            source[:, 1:] *= -1
            basket_controller.entry_orientation = _quat_multiply(
                _quat_multiply(state["object_state"][:, 3:7], source),
                basket_controller.entry_orientation,
            )
    grasp_planner = plan_grasp if basket_controller is None else basket_controller.plan_basket_grasp
    candidate_options = {}
    if args.recorded_grasp_attempt is not None:
        attempt = summary["planning"]["grasp_attempts"][args.recorded_grasp_attempt]
        candidate_options["candidate_orientations"] = limits.new_tensor(
            attempt["candidate_orientations_world_wxyz"]
        )
        candidate_options["pocket_depth_preference"] = attempt["pocket_depth_preference"]
        candidate_options["elevated_transit"] = attempt["approach_definition"].startswith(
            "elevated_cartesian"
        )
        args.realized_body = attempt["realized_body_ranking"]
        args.prefer_realized_body = attempt["prefer_realized_body"]
        args.minimum_pregrasp_standoff_m = attempt["minimum_pregrasp_standoff_m"]
        args.capture_axis_pitch_deg = math.degrees(attempt["capture_axis_pitch_rad"])
    if args.exclude_failed_attempts:
        failed = [
            attempt["candidate_orientations_world_wxyz"][0][attempt["selected"]]
            for attempt in summary.get("planning", {}).get("grasp_attempts", [])
            if attempt.get("execution_rejection") and attempt.get("selected") is not None
        ]
        candidate_options["excluded_orientations"] = limits.new_tensor(failed).reshape(-1, 4)
    if args.object_frame_orientations:
        from gr00t.rl.pick.planning import object_frame_grasp_bank

        candidate_options["candidate_orientations"] = object_frame_grasp_bank(
            state["object_state"][:, 3:7],
            episode["asset"].get("intrinsic_size_m", episode["asset"]["size_m"]),
        )
    paths, diagnostics = grasp_planner(
        state,
        state["object_state"][:, :3],
        episode["asset"].get("intrinsic_size_m", episode["asset"]["size_m"]),
        limits,
        collision,
        geometry,
        0.02,
        object_support_points=load_grasp_support(episode["asset"]["usd_path"], "cuda:0"),
        realized_body=args.realized_body,
        prefer_realized_body=args.prefer_realized_body,
        realized_body_tracking_weight=args.realized_body_tracking_weight,
        initial_object_height=initial_object_height,
        balanced_pitch_offsets_deg=args.balanced_pitch_offsets,
        wrist_band_rad=None if args.wrist_band_deg is None else math.radians(args.wrist_band_deg),
        preferred_orientation=None if preferred is None else limits.new_tensor(preferred)[None],
        refine_orientations=args.refine_orientations,
        natural_orientations=args.natural_orientations,
        minimum_pregrasp_standoff_m=args.minimum_pregrasp_standoff_m,
        capture_axis_pitch_rad=None
        if args.capture_axis_pitch_deg is None
        else math.radians(args.capture_axis_pitch_deg),
        retain_depth_candidates=args.retain_depth_candidates,
        **candidate_options,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics["diagnostic_surface_distances_only"] = args.surface_distances_only
    diagnostics["object_support_sampling"] = "all_hull_vertices_and_face_edge_samples"
    diagnostics["diagnostic_basket_obstacle_component"] = args.basket_obstacle_component
    diagnostics["diagnostic_object_frame_orientations"] = args.object_frame_orientations
    diagnostics["diagnostic_basket_final_entry"] = args.basket_final_entry
    if args.recorded_grasp_attempt is not None and paths is not None:
        p, q, _ = paths["approach"]
        start = args.frame % len(trace) + 1
        length = min(p.shape[1], len(trace) - start)
        if length:
            actual = trace[start : start + length]
            actual_p = torch.cat([row["ee_pos"] for row in actual]).to(limits.device)
            actual_q = torch.cat([row["ee_orn"] for row in actual]).to(limits.device)
            fixed, _ = collision._gripper_local_surfaces(p)
            command_z = (
                p[0, :length, None, 2] + _quat_rotate(q[0, :length, None], fixed[None])[..., 2]
            ).amin(-1)
            actual_z = (
                actual_p[:, None, 2] + _quat_rotate(actual_q[:, None], fixed[None])[..., 2]
            ).amin(-1)
            below = torch.where(actual_z < 0)[0]
            diagnostics["recorded_path_floor_tracking"] = {
                "definition": (
                    "reconstructed requested path versus recorded actual fixed-jaw geometry"
                ),
                "first_frame": start,
                "compared_frames": length,
                "command_min_z_m": float(command_z.min()),
                "actual_min_z_m": float(actual_z.min()),
                "maximum_downward_error_m": float((command_z - actual_z).max()),
                "first_actual_below_zero_frame": None if not len(below) else start + int(below[0]),
            }
    diagnostics["counterfactual_basket_local_xy"] = args.basket_local_xy
    diagnostics["counterfactual_object_local_rpy_deg"] = args.object_local_rpy_deg
    if args.audit_basket_colliders:
        if basket_controller is None:
            parser.error("--audit-basket-colliders requires a basket episode")
        quality = diagnostics["convex_contact_model"]
        mask = torch.tensor(quality["pocket_fit"], device=limits.device)
        mask &= torch.tensor(quality["antipodal_margin"], device=limits.device) >= 0
        mask &= torch.tensor(quality["support_pinch_compatible"], device=limits.device)
        indices = torch.where(mask[0])[0]
        q = limits.new_tensor(diagnostics["candidate_orientations_world_wxyz"])[:, indices]
        centers = limits.new_tensor(quality["command_center_local"])[:, indices]
        p = state["object_state"][:, None, :3] - _quat_rotate(q, centers)
        aperture = limits.new_tensor(quality["aperture_angle_rad"])[:, indices]
        world, faces = basket_controller.basket.world_mesh(state["root_states"][0])
        components = {}
        if len(indices):
            for index, name in enumerate(basket_controller.basket.names):
                surface = SceneGeometry(world, faces[faces[:, 0] // 8 == index])
                clearance = torch.minimum(
                    collision.gripper_clearance(surface, p, q),
                    collision.gripper_closing_clearance(surface, p, q, aperture),
                )[0]
                if bool((clearance < 0.006).any()):
                    components[name] = clearance.tolist()
        diagnostics["basket_collider_audit"] = {
            "candidate_indices": indices.tolist(),
            "blocking_collider_clearance_m": components,
            "note": "Selected depth of each pinch-compatible orientation; not all depths",
        }
        from gr00t.rl.pick.supported_asset_audit import gripper_collision_clouds

        fixed, closed = gripper_collision_clouds(str(limits.device))
        nominal_open = _quat_rotate(
            limits.new_tensor([math.cos(math.pi / 6), 0, -math.sin(math.pi / 6), 0]), closed
        )
        actual_fixed, actual_open = collision._gripper_local_surfaces(state["ee_pos"])
        diagnostics["basket_collider_audit"]["jaw_cloud_source_consistency_m"] = {
            name: float(torch.cdist(a[None], b[None]).amin(-1).amax())
            for name, a, b in (
                ("fixed", fixed, actual_fixed),
                ("open_moving", nominal_open, actual_open),
            )
        }
    if paths and args.audit_target_path:
        import numpy as np

        directory = Path(episode["asset"]["usd_path"]).parent
        surface = SceneGeometry(
            np.load(directory / "collision_support.npy"),
            np.load(directory / "collision_faces.npy"),
        )
        positions, orientations, _ = paths["approach"]
        inverse = state["object_state"][:, 3:7].clone()
        inverse[:, 1:] *= -1
        relative_positions = _quat_rotate(
            inverse[:, None], positions - state["object_state"][:, None, :3]
        )
        relative_orientations = _quat_multiply(inverse[:, None], orientations)
        clouds = torch.cat(collision._gripper_local_surfaces(positions))
        gap = surface.rigid_surface_clearance(
            relative_positions,
            relative_orientations,
            clouds,
            check_rays=False,
            signed_distance=True,
        )[0]
        blocked = torch.where(gap < -0.001)[0]
        diagnostics["target_approach_audit"] = {
            "definition": "Sampled signed jaw-to-original-object-hull gaps, not PhysX penetration",
            "minimum_signed_gap_m": float(gap.min()),
            "final_signed_gap_m": float(gap[-1]),
            "first_penetration_over_1mm_index": None if not len(blocked) else int(blocked[0]),
            "gap_by_path_point_m": gap.cpu().tolist(),
        }
    args.output.write_text(json.dumps(diagnostics, indent=2) + "\n")
    if paths:
        torch.save(paths, args.output.with_suffix(".pt"))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected": diagnostics["selected"],
                "feasible_count": sum(diagnostics["feasible"]),
            }
        )
    )


if __name__ == "__main__":
    main()
