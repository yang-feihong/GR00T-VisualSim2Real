"""Stationary-base basket grasp using the shared pick execution and retention gates."""

import math

import torch

from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.pick.basket import BASKET_GRASP_CLEARANCE_M, BasketGeometry
from gr00t.rl.pick.basket_entry import above_object_path
from gr00t.rl.pick.collision import ArmCollisionSamples
from gr00t.rl.pick.controller import PickController
from gr00t.rl.pick.stages import BasketStage
from gr00t.rl.pick.distractors import augment_scene_geometry, build_distractor_mesh
from gr00t.rl.pick.planning import (
    grasp_approach_origin,
    grasp_pose_bank,
    plan_grasp,
)
from gr00t.rl.pick.pocket import BASKET_MINIMUM_CENTER_DEPTH_M


def basket_vertical_clearance(basket, root, object_state, object_points, support_mask=None):
    world = _quat_rotate(root[:, None, 3:7], basket.boxes.reshape(-1, 3)) + root[:, None, :3]
    rim = world[..., 2].amax(-1)
    rotated = _quat_rotate(object_state[:, None, 3:7], object_points)
    z = rotated[..., 2]
    if support_mask is not None:
        z = z.masked_fill(~support_mask, torch.inf)
    bottom = object_state[:, 2] + z.amin(-1)
    return bottom - rim, rim


def basket_required_height(basket, state, support_points, lift_margin, support_mask=None):
    clearance, rim = basket_vertical_clearance(
        basket, state["root_states"], state["object_state"], support_points, support_mask
    )
    bottom_offset = state["object_state"][:, 2] - (clearance + rim)
    radii = support_points.norm(dim=-1)
    if support_mask is not None:
        radii = radii.masked_fill(~support_mask, -torch.inf)
    bottom_offset = torch.maximum(bottom_offset, radii.amax(-1))
    return rim + lift_margin + bottom_offset


def plan_basket_grasp_path(
    state,
    *args,
    minimum_height,
    final_entry=False,
    entry_orientation=None,
    maximum_tool_tilt_deg=30.0,
    **kwargs,
):
    """Shared full-path admission and execution planning, with identical geometry gates."""
    target, object_size = args[:2]
    root = state["root_states"]
    heading = torch.atan2(target[:, 1] - root[:, 1], target[:, 0] - root[:, 0])
    orientation = kwargs.pop("candidate_orientations", None)
    if final_entry:
        orientation = entry_orientation[:, None]
    if orientation is None:
        _, orientation, _ = grasp_pose_bank(
            target,
            heading,
            object_size,
            grasp_approach_origin(root),
            reference_orientation=state["ee_orn"],
        )
        orientation = torch.cat((state["ee_orn"][:, None], orientation), 1)
    axis = _quat_rotate(orientation, target.new_tensor([1, 0, 0]))
    tilt = float(maximum_tool_tilt_deg)
    orientation = orientation[:, axis[0, :, 2] <= -math.cos(math.radians(tilt))]
    if orientation.shape[1] == 0:
        return None, {
            "failure": "no_grasp_orientation_inside_top_down_cone",
            "required_object_lift_m": 0.10,
            "maximum_tool_tilt_deg": tilt,
            "candidate_count_before_tilt": axis.shape[1],
            "minimum_object_height_world_m": float(minimum_height[0]),
        }
    kwargs.update(
        candidate_orientations=orientation,
        preferred_orientation=None,
        minimum_object_height_m=float(minimum_height[0]),
        withdraw_distance_m=0.0,
        require_precise_carry=False,
        validate_carry_preview=False,
        require_safe_arm_path=True,
        collider_clearance_m=BASKET_GRASP_CLEARANCE_M,
        support_normal_world=_quat_rotate(root[:, 3:7], target.new_tensor([[0, 0, 1]])),
        prefer_realized_body=True,
        object_frame_fallback=True,
        fallback_tool_tilt_deg=tilt,
        retain_depth_candidates=True,
        minimum_center_depth_m=BASKET_MINIMUM_CENTER_DEPTH_M,
    )
    kwargs.setdefault("wrist_motion_weight", 1e-5)
    return plan_grasp(state, *args, **kwargs)


def basket_planning_geometry(scene, basket, episode, state):
    obstacles = build_distractor_mesh(episode.get("distractors", []), state["distractor_states"])
    nonrobot = augment_scene_geometry(scene, obstacles)
    world, faces = basket.world_mesh(state["root_states"][0])
    result = augment_scene_geometry(nonrobot, {"points": world, "faces": faces})
    result.nonrobot_geometry = nonrobot
    result.basket_model = basket
    return result


class BasketArmCollisionSamples:
    """Self-collision is invariant to hypothetical trunk heights and world poses."""

    def __init__(self, arm):
        self.arm = arm

    def __getattr__(self, name):
        return getattr(self.arm, name)

    def clearance(self, geometry, roots, joints, **kwargs):
        scene_clear = self.arm.clearance(geometry.nonrobot_geometry, roots, joints, **kwargs)
        local = torch.zeros_like(roots)
        local[..., 3] = 1
        self_clear = self.arm.clearance(geometry.basket_model.geometry, local, joints, **kwargs)
        return torch.minimum(scene_clear, self_clear)


class BasketPickController(PickController):
    Stage = BasketStage

    def __init__(self, env, episode, geometry, **kwargs):
        self.nudge_attempts = 0
        self.final_entry = False
        self.awaiting_entry_refresh = False
        self.entry_orientation = None
        self.entry_object_orientation = None
        self.entry_refresh_count = 0
        self.initial_grasp_plan = getattr(geometry, "initial_grasp_plan", None)
        self.basket = getattr(geometry, "basket_model", None) or BasketGeometry.from_robot_usd(
            device=env.device
        )
        self.lift_margin = float(episode.get("lift_above_rim_m", 0.10))
        kwargs["collision"] = BasketArmCollisionSamples(
            kwargs.get("collision") or ArmCollisionSamples(env)
        )
        super().__init__(
            env,
            episode,
            geometry,
            **kwargs,
        )

    def required_height(self, state):
        return basket_required_height(
            self.basket, state, self.object_support_points, self.lift_margin
        )

    def plan_basket_grasp(self, state, *args, **kwargs):
        tilt = float(getattr(self, "maximum_tool_tilt_deg", 30.0))
        cache = getattr(self, "initial_grasp_plan", None)
        self.initial_grasp_plan = None
        reusable = (
            cache is not None
            and not self.final_entry
            and kwargs.get("candidate_orientations") is None
            and cache["maximum_tool_tilt_deg"] == tilt
            and all(torch.equal(state[key], value) for key, value in cache["state"].items())
        )
        if reusable:
            paths, diagnostics = cache["paths"], cache["diagnostics"]
            diagnostics["reused_initial_admission_plan"] = True
        else:
            paths, diagnostics = plan_basket_grasp_path(
                state,
                *args,
                minimum_height=self.required_height(state),
                final_entry=self.final_entry,
                entry_orientation=getattr(self, "entry_orientation", None),
                maximum_tool_tilt_deg=tilt,
                **kwargs,
            )
        diagnostics["task"] = "stationary_basket_top_down"
        diagnostics["lift_above_rim_m"] = self.lift_margin
        diagnostics["maximum_tool_tilt_deg"] = tilt
        diagnostics["entry_leg"] = (
            "insert_at_measured_object_pose" if self.final_entry else "above_object"
        )
        if paths is not None and not self.final_entry:
            paths["approach"], self.entry_orientation = above_object_path(
                paths["approach"], diagnostics
            )
            self.awaiting_entry_refresh = True
            self.entry_object_orientation = state["object_state"][:, 3:7].clone()
        return paths, diagnostics

    def plan_basket_carry(self, state, *args, **kwargs):
        from gr00t.rl.pick.basket_carry import plan_basket_carry_bank

        _, initial_height, aperture, limits, collision, geometry, dt = args
        options = (
            {}
            if not hasattr(self, "carry_tilt_angles_deg")
            else {"tilt_angles_deg": self.carry_tilt_angles_deg}
        )
        return plan_basket_carry_bank(
            state,
            initial_height,
            aperture,
            limits,
            collision,
            geometry,
            dt,
            basket=self.basket,
            support_points=self.object_support_points,
            minimum_object_height=self.required_height(state),
            **options,
        )
