"""Passive per-episode collector for the AutoOpen physical data standard."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import torch

from gr00t.rl.auto_open.control import _through_tail_clearance
from gr00t.rl.envs.door.b2z1_reset_geometry import (
    b2z1_gripper_pose_and_jacobian,
)

from .auto_open_standard import (
    B2_JOINT_NAMES,
    CONTROL_HZ,
    TERMINAL_ARM_JOINT_SPEED_RAD_S,
    dataset_tool_quaternion_from_gripper_quaternion,
    dataset_tool_pose_from_gripper_pose,
    height_invariant_pose,
    pack_action,
    pack_observation_state,
    quaternion_wxyz_to_matrix,
    transform_dataset_images,
)
from .auto_open_writer import (
    CAMERA_SHAPES,
    AutoOpenEpisodeData,
    StreamingVideoEncoder,
)
from .contact_audit import SubstepContactAudit


HANDLE_PRESS_STAGES = {
    "PREGRASP",
    "OPEN_GRIPPER",
    "CENTER_HANDLE",
    "GRASP_HANDLE",
    "TURN_HANDLE",
    "RETURN_HANDLE",
    "OPEN_AND_CLEAR",
    "RETREAT_ARM",
    "RESTORE_GRIPPER",
}

MIDDLE_PRESET_POSITION_TOLERANCE_RAD = 0.05
MIDDLE_PRESET_STABLE_FRAMES = 10


def _numpy(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _phase_for_stage(stage_name: str) -> str:
    if stage_name == "APPROACH":
        return "approach"
    if stage_name in HANDLE_PRESS_STAGES:
        return "handle_press"
    if stage_name in {"SUCCESS", "terminal_hold"}:
        return "terminal_hold"
    return "traversal"


def _door_edge_gap_m(hinge_angle, hinge_to_free_edge, panel_half_thickness):
    """Conservative free-edge distance from the closed door plane."""
    angle = abs(float(hinge_angle))
    reach = float(hinge_to_free_edge)
    thickness = 2.0 * float(panel_half_thickness)
    return max(
        reach * np.sin(angle) - thickness * (1.0 - np.cos(angle)),
        0.0,
    )


class AutoOpenStandardEpisodeCollector:
    """Record one environment without participating in AutoOpen decisions."""

    def __init__(
        self,
        runner,
        controller,
        *,
        episode_index: int,
        seed: int,
        video_staging_root: Path | None = None,
    ):
        if int(runner.env.num_envs) != 1:
            raise ValueError("the initial standard collector requires one env per shard")
        self.runner = runner
        self.env = runner.env
        self.controller = controller
        self.episode_index = int(episode_index)
        self.seed = int(seed)
        self.contact_audit = SubstepContactAudit.create(self.env)
        self.contact_audit.attach()
        self.states = []
        self.actions = []
        self.images = defaultdict(list)
        self.video_encoders = {}
        self.video_staging_root = (
            None if video_staging_root is None else Path(video_staging_root)
        )
        self.audit = defaultdict(list)
        self.contacts = []
        self.events = {}
        self.timing = defaultdict(float)
        self._pending = None
        self._previous_stage = None
        self._previous_arm_position = None
        self._middle_preset_interval = False
        self._middle_preset_reset_active = False
        self._middle_preset_reset_complete = False
        self._middle_preset_stable_frames = 0

        dof_names = list(self.env.dof_names)
        self.arm_indices = np.asarray(
            [dof_names.index(name) for name in self.env.arm_joint_names], dtype=np.int64
        )
        self.b2_indices = np.asarray(
            [dof_names.index(name) for name in B2_JOINT_NAMES], dtype=np.int64
        )
        self.gripper_index = int(_numpy(self.env.gripper_joint_ids).reshape(-1)[0])
        body_names = list(self.env.body_names)
        self.foot_indices = np.asarray(
            [body_names.index(name) for name in self.env.cfg.foot_body_names],
            dtype=np.int64,
        )

    @property
    def contact_view_description(self):
        return self.contact_audit.description

    def close(self):
        self.contact_audit.detach()
        for encoder in self.video_encoders.values():
            encoder.close()

    def capture_before_step(self, state, high_level_action, *, stage_name=None):
        """Latch observation/image at t[k] immediately before executing action[k]."""
        if self._pending is not None:
            raise RuntimeError("previous standard-data frame has not been completed")
        stage_name = stage_name or self.controller.stage.name
        frame_index = len(self.states)
        sim_time_ns = int(round(frame_index * 20_000_000))
        phase_start = time.perf_counter()
        rgb_frames = self._read_current_images()
        self.timing["image_read_s"] += time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        images = transform_dataset_images(
            {
                "base_0_rgb": _numpy(rgb_frames["base_0_rgb"]),
                "left_wrist_0_rgb": _numpy(rgb_frames["left_wrist_0_rgb"]),
            }
        )
        self.timing["image_transform_s"] += time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        for name, image in images.items():
            if self.video_staging_root is None:
                self.images[name].append(image)
                continue
            encoder = self.video_encoders.get(name)
            if encoder is None:
                encoder = StreamingVideoEncoder(
                    self.video_staging_root / name / "episode.mp4",
                    CAMERA_SHAPES[name],
                )
                self.video_encoders[name] = encoder
            encoder.append(image)
        self.timing["video_enqueue_wait_s"] += time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        snapshot = self._snapshot_state(state)
        self.states.append(snapshot["packed_state"])
        self._append_state_audit(snapshot, frame_index, sim_time_ns, stage_name)
        self.contact_audit.begin_tick()
        self.timing["state_snapshot_s"] += time.perf_counter() - phase_start
        self._pending = {
            "state": state,
            "snapshot": snapshot,
            "high": np.asarray(high_level_action, dtype=np.float32).copy(),
            "stage_name": stage_name,
            "frame_index": frame_index,
            "sim_time_ns": sim_time_ns,
        }

    def _read_current_images(self):
        """Read both 50 Hz sensors once per integer control tick.

        The generic preview publisher uses floating-point period comparisons,
        which can skip a nominal 20 ms instant such as 60 ms.  Dataset timing
        is an integer tick clock, so it reads the already-updated sensor data
        directly and rejects any missing modality.
        """
        manager = self.runner.rgb_camera_manager
        if manager is None or not manager.enabled:
            raise RuntimeError("standard data collection requires an RGB camera manager")
        images_by_name = manager.render_images(
            env_id=0,
            rig_names=("base_0_rgb", "left_wrist_0_rgb"),
            update=False,
        )
        frames = {}
        for name in ("base_0_rgb", "left_wrist_0_rgb"):
            images = images_by_name.get(name, ())
            if len(images) != 1:
                raise RuntimeError(
                    f"camera {name} returned {len(images)} frames at dataset tick"
                )
            frames[name] = images[0]
        return frames

    def complete_after_step(self, result):
        """Attach the target actually dispatched over [t[k], t[k+1])."""
        if self._pending is None:
            raise RuntimeError("capture_before_step must precede complete_after_step")
        pending = self._pending
        phase_start = time.perf_counter()
        post_state = result["state"]
        arm_target = _numpy(post_state["arm_position_target"])[0].astype(np.float32)
        arm_raw_target = _numpy(post_state["arm_raw_position_target"])[0].astype(
            np.float32
        )
        base_dispatched = _numpy(post_state["base_command"])[0].astype(np.float32)
        gripper_dispatched = float(_numpy(post_state["gripper_command"])[0, 0])
        gripper_actuator_target = float(
            _numpy(post_state["gripper_actuator_target"])[0, 0]
        )
        inactive, reset = self._mode_flags(pending["snapshot"], arm_raw_target)
        target_pose = self._target_height_invariant_pose(
            pending["snapshot"], arm_target
        )
        packed_action = pack_action(
            base_command=base_dispatched,
            arm_teleop_inactive=inactive,
            arm_reset=reset,
            control_height_invariant_ee=target_pose,
            gripper_target=gripper_dispatched,
        )
        self.actions.append(packed_action)
        self.timing["action_pack_s"] += time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        contacts = self.contact_audit.end_tick()
        max_impulse = max((row.max_normal_impulse for row in contacts), default=0.0)
        max_force = max((row.max_contact_force for row in contacts), default=0.0)
        max_penetration = max(
            (row.max_penetration_depth for row in contacts), default=0.0
        )
        self.audit["max_normal_impulse"].append(max_impulse)
        self.audit["max_contact_force"].append(max_force)
        self.audit["max_penetration_depth"].append(max_penetration)
        self.audit["ccd_triggered"].append(False)
        self.audit["base_command_pre"].append(pending["high"][:3])
        self.audit["base_command_dispatched"].append(base_dispatched)
        self.audit["arm_joint_target_pre"].append(arm_raw_target)
        self.audit["arm_joint_target_dispatched"].append(arm_target)
        self.audit["arm_joint_position_target"].append(arm_target)
        self.audit["gripper_target_pre"].append(float(pending["high"][9]))
        self.audit["gripper_target_dispatched"].append(gripper_dispatched)
        self.audit["gripper_actuator_target"].append(gripper_actuator_target)
        self.audit["arm_teleop_inactive_pre"].append(inactive)
        self.audit["arm_teleop_inactive_dispatched"].append(inactive)
        self.audit["arm_reset_pre"].append(reset)
        self.audit["arm_reset_dispatched"].append(reset)
        for row in contacts:
            self.contacts.append(
                {
                    "episode_index": self.episode_index,
                    "frame_index": pending["frame_index"],
                    "sim_time_ns": pending["sim_time_ns"],
                    **row.__dict__,
                }
            )
        self._record_events(post_state)
        self.timing["contact_finalize_s"] += time.perf_counter() - phase_start
        self._pending = None

    def finish(self, acceptance_reasons=(), provenance=None):
        if self._pending is not None:
            raise RuntimeError("cannot finish with an incomplete control frame")
        for encoder in self.video_encoders.values():
            encoder.close()
        encode_times = [
            float(encoder.encode_wall_s) for encoder in self.video_encoders.values()
        ]
        self.timing["video_encode_worker_s_sum"] = sum(encode_times)
        self.timing["video_encode_worker_s_max"] = max(encode_times, default=0.0)
        return AutoOpenEpisodeData(
            episode_index=self.episode_index,
            seed=self.seed,
            observation_state=np.asarray(self.states, dtype=np.float32),
            action=np.asarray(self.actions, dtype=np.float32),
            images={name: tuple(frames) for name, frames in self.images.items()},
            raw_audit={name: np.asarray(values) for name, values in self.audit.items()},
            staged_videos={
                name: encoder.path for name, encoder in self.video_encoders.items()
            },
            image_frame_counts={
                name: encoder.frame_count for name, encoder in self.video_encoders.items()
            },
            raw_contacts=tuple(self.contacts),
            event_frames=dict(self.events),
            acceptance_reasons=tuple(acceptance_reasons),
            provenance=dict(provenance or {}),
        )

    def _snapshot_state(self, state):
        root = _numpy(state["root_states"])[0]
        dof_pos = _numpy(state["dof_pos"])[0]
        dof_vel = _numpy(state["dof_vel"])[0]
        ee_position = _numpy(state["ee_pos"])[0]
        ee_quaternion = _numpy(state["ee_orn"])[0]
        ee_rotation = quaternion_wxyz_to_matrix(ee_quaternion)
        tool_position, tool_rotation = dataset_tool_pose_from_gripper_pose(
            ee_position, ee_rotation
        )
        tool_quaternion = dataset_tool_quaternion_from_gripper_quaternion(
            ee_quaternion
        )
        body_state = _numpy(state["robot_body_state_w"])[0]
        contact_forces = _numpy(state["contact_forces_w"])[0]
        foot_forces = contact_forces[self.foot_indices]
        foot_normal_force = np.maximum(foot_forces[:, 2], 0.0)
        door_q = _numpy(state["door_joint_position"])[0]
        door_qd = _numpy(state["door_joint_velocity"])[0]
        arm_position = np.asarray(dof_pos[self.arm_indices], dtype=np.float64)
        if self._previous_arm_position is None:
            arm_velocity = np.zeros_like(arm_position)
        else:
            arm_velocity = (arm_position - self._previous_arm_position) * CONTROL_HZ
        self._previous_arm_position = arm_position.copy()
        packed_state = pack_observation_state(
            arm_position=arm_position,
            arm_velocity=arm_velocity,
            gripper_feedback=dof_pos[self.gripper_index],
            b2_position=dof_pos[self.b2_indices],
            b2_velocity=dof_vel[self.b2_indices],
            root_position_world=root[:3],
            root_quaternion_wxyz=root[3:7],
            root_linear_velocity_world=root[7:10],
            root_angular_velocity_world=root[10:13],
            actual_tool_position_world=tool_position,
            actual_tool_rotation_world=tool_rotation,
        )
        return {
            "packed_state": packed_state,
            "root": root,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "arm_velocity": arm_velocity,
            "ee_position": ee_position,
            "ee_quaternion": ee_quaternion,
            "tool_position": tool_position,
            "tool_rotation": tool_rotation,
            "tool_quaternion": tool_quaternion,
            "body_state": body_state,
            "foot_forces": foot_forces,
            "foot_normal_force": foot_normal_force,
            "door_q": door_q,
            "door_qd": door_qd,
        }

    def _append_state_audit(self, value, frame_index, sim_time_ns, stage_name):
        controller = self.controller
        self.audit["episode_index"].append(self.episode_index)
        self.audit["frame_index"].append(frame_index)
        self.audit["sim_time_ns"].append(sim_time_ns)
        self.audit["base_image_capture_time_ns"].append(sim_time_ns)
        self.audit["wrist_image_capture_time_ns"].append(sim_time_ns)
        self.audit["root_position_world"].append(value["root"][:3])
        self.audit["root_quaternion_xyzw"].append(value["root"][[4, 5, 6, 3]])
        self.audit["root_linear_velocity_world"].append(value["root"][7:10])
        self.audit["root_angular_velocity_world"].append(value["root"][10:13])
        self.audit["arm_joint_position"].append(value["dof_pos"][self.arm_indices])
        self.audit["arm_joint_velocity"].append(value["arm_velocity"])
        self.audit["physx_arm_joint_velocity"].append(
            value["dof_vel"][self.arm_indices]
        )
        self.audit["gripper_feedback"].append(value["dof_pos"][self.gripper_index])
        self.audit["b2_joint_position"].append(value["dof_pos"][self.b2_indices])
        self.audit["b2_joint_velocity"].append(value["dof_vel"][self.b2_indices])
        self.audit["foot_position_world"].append(
            value["body_state"][self.foot_indices, :3].reshape(-1)
        )
        self.audit["foot_contact"].append(value["foot_normal_force"] > 1.0)
        self.audit["foot_normal_force"].append(value["foot_normal_force"])
        self.audit["ee_pose_world_xyzw"].append(
            np.concatenate(
                (value["tool_position"], value["tool_quaternion"][[1, 2, 3, 0]])
            )
        )
        self.audit["door_hinge_position"].append(value["door_q"][controller.hinge_idx])
        self.audit["door_hinge_velocity"].append(value["door_qd"][controller.hinge_idx])
        hinge_angle = float(value["door_q"][controller.hinge_idx])
        self.audit["door_edge_gap_m"].append(
            _door_edge_gap_m(
                hinge_angle,
                controller.panel_hinge_to_free_edge,
                controller.panel_half_thickness,
            )
        )
        self.audit["handle_position"].append(value["door_q"][controller.handle_joint_idx])
        self.audit["handle_velocity"].append(value["door_qd"][controller.handle_joint_idx])
        latch = float(value["door_q"][controller.latch_joint_idx])
        self.audit["door_latch_position"].append(latch)
        self.audit["door_latch_state"].append("unlocked" if latch >= 0.02 else "locked")
        self.audit["robot_clear_of_door"].append(
            bool(
                _through_tail_clearance(
                    controller, self._pending_state_view(value)
                )
                > 0.0
            )
        )
        self.audit["autoopen_stage"].append(stage_name)
        self.audit["physical_phase"].append(_phase_for_stage(stage_name))

    @staticmethod
    def _pending_state_view(snapshot):
        return {"root_states": snapshot["root"][None, :]}

    def _mode_flags(self, snapshot, arm_raw_target):
        joint_mode = bool(_numpy(self.env.teleop_arm_joint_control_mask)[0])
        if not joint_mode:
            self._clear_middle_preset_mode()
            return False, False

        middle_target = np.asarray(
            self.controller.middle_arm_joint_target, dtype=np.float32
        )
        returning_to_middle = np.allclose(
            np.asarray(arm_raw_target, dtype=np.float32),
            middle_target,
            rtol=0.0,
            atol=1.0e-6,
        )
        if not returning_to_middle:
            self._clear_middle_preset_mode()
            return True, False

        arm_position = np.asarray(snapshot["dof_pos"])[self.arm_indices]
        arm_velocity = np.asarray(snapshot["arm_velocity"])
        middle_reached = bool(
            np.max(np.abs(arm_position - middle_target))
            < MIDDLE_PRESET_POSITION_TOLERANCE_RAD
            and np.max(np.abs(arm_velocity))
            < TERMINAL_ARM_JOINT_SPEED_RAD_S
        )
        if not self._middle_preset_interval:
            self._middle_preset_interval = True
            self._middle_preset_reset_active = not middle_reached
            self._middle_preset_reset_complete = middle_reached
            self._middle_preset_stable_frames = 0

        if self._middle_preset_reset_complete:
            return True, False

        self._middle_preset_stable_frames = (
            self._middle_preset_stable_frames + 1 if middle_reached else 0
        )
        if self._middle_preset_stable_frames >= MIDDLE_PRESET_STABLE_FRAMES:
            self._middle_preset_reset_active = False
            self._middle_preset_reset_complete = True
            return True, False
        return False, True

    def _clear_middle_preset_mode(self):
        self._middle_preset_interval = False
        self._middle_preset_reset_active = False
        self._middle_preset_reset_complete = False
        self._middle_preset_stable_frames = 0

    def _target_height_invariant_pose(self, snapshot, arm_target):
        root_position = torch.as_tensor(snapshot["root"][:3], dtype=torch.float64)[None]
        root_quaternion_xyzw = torch.as_tensor(
            snapshot["root"][[4, 5, 6, 3]], dtype=torch.float64
        )[None]
        arm = torch.as_tensor(arm_target, dtype=torch.float64)[None]
        position, rotation, _ = b2z1_gripper_pose_and_jacobian(
            root_position, root_quaternion_xyzw, arm
        )
        tool_position, tool_rotation = dataset_tool_pose_from_gripper_pose(
            position[0].numpy(), rotation[0].numpy()
        )

        return height_invariant_pose(
            snapshot["root"][:3],
            snapshot["root"][3:7],
            tool_position,
            tool_rotation,
        )

    def _record_events(self, state):
        frame = len(self.actions) - 1
        joints = _numpy(state["door_joint_position"])[0]
        if (
            "handle_unlocked" not in self.events
            and float(joints[self.controller.latch_joint_idx]) >= 0.02
        ):
            self.events["handle_unlocked"] = frame
        if (
            "door_gap_20mm" not in self.events
            and float(self.audit["door_edge_gap_m"][-1]) >= 0.02
        ):
            self.events["door_gap_20mm"] = frame
        if "robot_clear" not in self.events and bool(self.audit["robot_clear_of_door"][-1]):
            self.events["robot_clear"] = frame
