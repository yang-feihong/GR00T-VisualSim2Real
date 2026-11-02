"""Pick configuration for the shared batched FSM, without a scalar stage runtime."""

import torch

from gr00t.rl.auto_open.arm_model import read_arm_joint_limits
from gr00t.rl.auto_open.base_trajectory import BasePoseTrajectory
from gr00t.rl.auto_open.control import APPROACH_ARRIVAL
from gr00t.rl.auto_open.execution import StageExecutionMixin
from gr00t.rl.auto_open.motion_targets import PaddedPathBank
from gr00t.rl.pick.collision import ArmCollisionSamples
from gr00t.rl.pick.gaze import sample_approach_offsets
from gr00t.rl.pick.stages import PickStage
from gr00t.rl.utils.b2z1_asset import ROBOT_URDF


def pose_window_speed(positions, dt):
    time = torch.arange(positions.shape[1], device=positions.device, dtype=positions.dtype)
    time = time - (positions.shape[1] - 1) / 2
    weights = time / (dt * time.square().sum())
    displacement = positions - positions[:, :1]
    return (displacement * weights[None, :, None]).sum(1).norm(dim=-1)


class PickController(StageExecutionMixin):
    Stage = PickStage

    def __init__(
        self,
        env,
        episode,
        geometry,
        *,
        collision=None,
        object_support_points=None,
        initialize=True,
    ):
        self.env, self.episode, self.geometry = env, episode, geometry
        self.object_support_points = object_support_points
        if episode.get("support_type") == "floor":
            from gr00t.rl.pick.measured_station import MeasuredLowReachPlanner

            self.measured_reach = MeasuredLowReachPlanner(
                episode, getattr(env, "pick_policy_checkpoint_sha256", None)
            )
        self.device, self.num_envs = env.device, env.num_envs
        self.rows = torch.arange(self.num_envs, device=self.device)
        self.stage = torch.full(
            (self.num_envs,), int(PickStage.APPROACH_OBJECT), device=self.device
        )
        self.current_stage = PickStage.APPROACH_OBJECT
        self.reason = None
        self.planning_diagnostics = {}
        self.grasp_retries = 0
        self.tried_grasp_orientations = torch.empty(0, 4, device=self.device)
        self.target = torch.tensor(episode["object_position"], device=self.device)[None]
        self.arm_induced_pitch = torch.full(
            (self.num_envs,), env.arm_induced_pitch, device=self.device
        )
        self.base_command_previous = torch.zeros(self.num_envs, 3, device=self.device)
        for name in ("pending_sign", "pending_steps", "lock_steps"):
            setattr(
                self,
                "base_reversal_" + name,
                torch.zeros(
                    self.num_envs,
                    3,
                    dtype=torch.int8 if name == "pending_sign" else torch.int32,
                    device=self.device,
                ),
            )
        for name in ("base_arrival_confirmed", "base_stability_confirmed"):
            setattr(self, name, torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        for name in (
            "base_arrival_settle_steps",
            "base_arrival_max_settle_steps",
            "base_stability_steps",
            "base_arrival_required_frames",
        ):
            setattr(self, name, torch.zeros(self.num_envs, dtype=torch.int32, device=self.device))
        for name in (
            "inner_distance",
            "outer_distance",
            "inner_yaw",
            "outer_yaw",
            "max_linear_speed",
            "max_angular_speed",
        ):
            setattr(self, "base_arrival_" + name, torch.zeros(self.num_envs, device=self.device))
        self.through_min_base_height = torch.full((self.num_envs,), 0.18, device=self.device)
        self.reset_base_arrival(self.rows, APPROACH_ARRIVAL)
        self.min_grip_force_n = float(getattr(env, "pick_min_grip_force_n", 0.3))
        self.min_lift_m = 0.10
        sensor = getattr(env, "contact_sensor", None)
        names = list(getattr(sensor, "body_names", []))
        self.contact_body_names = names
        self.nonfoot_contact_ids = torch.tensor(
            [i for i, name in enumerate(names) if not name.lower().endswith("foot")],
            device=self.device,
            dtype=torch.int64,
        )
        self.jaw_contact_ids = [
            next((i for i, name in enumerate(names) if name.split("/")[-1].lower() == jaw), None)
            for jaw in ("link06", "grippermover")
        ]
        self.arm_path = PaddedPathBank(self.num_envs, 4096, self.device)
        self.base_path = BasePoseTrajectory(self.num_envs, self.device)
        self.collision = ArmCollisionSamples(env) if collision is None else collision
        self.limits = getattr(env, "pick_arm_joint_limits", None)
        if self.limits is None:
            self.limits = torch.tensor(
                read_arm_joint_limits(ROBOT_URDF).copy(), device=self.device, dtype=torch.float32
            )
        self.paths = None
        self.grasp_object_relative = None
        self.initial_grasp_orientation = None
        self.initial_grasp_joint_reference = None
        state = env.get_control_state()
        self.initial_object_z = state["object_state"][:, 2].clone()
        self.hold_xy = state["root_states"][:, :2].clone()
        self.hold_yaw = torch.full(
            (self.num_envs,),
            episode["base_yaw"],
            device=self.device,
            dtype=state["root_states"].dtype,
        )
        self.initial_joints = state["dof_pos"][:, env.arm_joint_ids].clone()
        self.approach_gaze_valid = True
        self.install_arm_path(state["ee_pos"][:, None], state["ee_orn"][:, None])
        if initialize and episode.get("task_type") != "pick_basket":
            self.deferred_initial_station = True
            self.approach_random_offsets = getattr(env, "pick_approach_offsets", None)
            if self.approach_random_offsets is None:
                self.approach_random_offsets = sample_approach_offsets(
                    episode["seed"], self.device
                )[None]
            self.gaze_offsets = self.approach_random_offsets[:, 1:]

    def install_arm_path(self, positions, orientations, lengths=None):
        self.arm_path.install_tensor(
            self.rows,
            positions,
            orientations,
            torch.full((self.num_envs,), positions.shape[1], device=self.device)
            if lengths is None
            else lengths,
        )
