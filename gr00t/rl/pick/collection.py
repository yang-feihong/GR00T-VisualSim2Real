"""GPU packing of the shared 58-state/25-action collection contract."""

import torch

from gr00t.rl.envs.door.b2z1_reset_geometry import (
    b2z1_gripper_pose_and_jacobian,
    quaternion_xyzw_to_matrix,
)


class InitializationDebugRecorder:
    """Video-only admission observer; never publishes commands or steps physics."""

    def __init__(self, runner, seeds, output, task_kind):
        from gr00t.rl.data_collection.auto_open_standard import camera_streams_for_mode

        self.manager = runner.rgb_camera_manager
        self.seeds, self.output = tuple(seeds), output
        self.names = tuple(camera_streams_for_mode("six", task_kind=task_kind))
        self.encoders = {}
        self.frames = []

    def capture(self, cohort, step):
        if step % 10:
            return
        from gr00t.rl.data_collection.auto_open_writer import (
            StreamingVideoEncoder,
            compose_six_camera_mosaic,
        )

        images = self.manager.render_images_batch(
            list(range(len(self.seeds))), rig_names=self.names
        )
        for row, seed in enumerate(self.seeds):
            if row not in self.encoders:
                self.encoders[row] = StreamingVideoEncoder(
                    self.output / f"seed_{seed}/initialization_six_camera_mosaic.mp4",
                    (720, 1440, 3),
                )
            self.encoders[row].append(
                compose_six_camera_mosaic(
                    (f"INIT {cohort}:{step} {name}", images[row][name][0]) for name in self.names
                )
            )
        self.frames.append({"cohort": cohort, "cohort_step": step})

    def close(self):
        import json

        for encoder in self.encoders.values():
            encoder.close()
        (self.output / "initialization_video_frames.json").write_text(
            json.dumps({"sampling_stride": 10, "playback_speed": 10, "frames": self.frames}) + "\n"
        )


def _yaw_frame_pose(root, yaw, position, rotation):
    c, s = yaw.cos(), yaw.sin()
    world_to_yaw = torch.stack((c, s, c * 0, -s, c, c * 0, c * 0, c * 0, c * 0 + 1), -1).reshape(
        -1, 3, 3
    )
    translation = root[:, :3].clone()
    translation[:, 2] = 0
    local = torch.bmm(world_to_yaw, (position - translation)[..., None]).squeeze(-1)
    local -= root.new_tensor([0.22, 0.0, 0.7])
    gripper_to_dataset = root.new_tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation = world_to_yaw @ rotation @ gripper_to_dataset
    return torch.cat((rotation[:, :, 0], rotation[:, :, 1], local), -1)


def pack_control_frame(
    before,
    after,
    *,
    arm_ids,
    b2_ids,
    gripper_id,
    previous_arm_position=None,
    joint_mode=None,
    dt=0.02,
):
    """Observe at t[k]; record the action actually dispatched during [k,k+1)."""
    root = before["root_states"]
    root_rotation = quaternion_xyzw_to_matrix(root[:, [4, 5, 6, 3]])
    roll = torch.atan2(root_rotation[:, 2, 1], root_rotation[:, 2, 2])
    pitch = torch.atan2(
        -root_rotation[:, 2, 0], torch.hypot(root_rotation[:, 2, 1], root_rotation[:, 2, 2])
    )
    yaw = torch.atan2(root_rotation[:, 1, 0], root_rotation[:, 0, 0])
    arm_position = before["dof_pos"][:, arm_ids]
    arm_velocity = (
        torch.zeros_like(arm_position)
        if previous_arm_position is None
        else (arm_position - previous_arm_position) / dt
    )
    ee_rotation = quaternion_xyzw_to_matrix(before["ee_orn"][:, [1, 2, 3, 0]])
    actual_pose = _yaw_frame_pose(root, yaw, before["ee_pos"], ee_rotation)
    body_velocity = torch.bmm(root_rotation.transpose(-1, -2), root[:, 7:10, None]).squeeze(-1)
    body_angular = torch.bmm(root_rotation.transpose(-1, -2), root[:, 10:13, None]).squeeze(-1)
    state = torch.cat(
        (
            arm_position,
            arm_velocity,
            before["dof_pos"][:, gripper_id : gripper_id + 1],
            before["dof_pos"][:, b2_ids],
            before["dof_vel"][:, b2_ids],
            torch.stack((roll, pitch, root[:, 2], root[:, 0], root[:, 1], yaw), -1),
            body_velocity,
            body_angular,
            actual_pose,
        ),
        -1,
    )
    target_position, target_rotation, _ = b2z1_gripper_pose_and_jacobian(
        root[:, :3], root[:, [4, 5, 6, 3]], after["arm_position_target"]
    )
    control_pose = _yaw_frame_pose(root, yaw, target_position, target_rotation)
    flags = root.new_zeros(len(root), 2)
    if joint_mode is not None:
        flags[:, 0] = joint_mode
    action = torch.cat(
        (
            after["base_command"],
            flags,
            control_pose,
            after["gripper_command"].reshape(-1, 1),
            root.new_zeros(len(root), 1),
            control_pose,
        ),
        -1,
    )
    return state, action


class PickEpisodeCollector:
    """Passive GPU frame capture; only contact audit/serialization uses the CPU."""

    PRE_KEYS = (
        "root_states",
        "dof_pos",
        "dof_vel",
        "ee_pos",
        "ee_orn",
        "object_state",
        "distractor_states",
    )
    POST_KEYS = PRE_KEYS + (
        "arm_position_target",
        "base_command",
        "gripper_command",
        "object_finger_forces",
        "object_finger_friction_forces",
        "object_finger_contact_points",
        "contact_forces_w",
    )

    DEBUG_POST_KEYS = (
        "ee_intended_goal_world",
        "ee_command_height_offset_m",
        "lowlevel_policy_observation",
        "lowlevel_policy_action",
        "physics_substep_dof_pos",
        "physics_substep_dof_vel",
        "physics_substep_torque",
        "physics_substep_position_target",
        "ee_goal_cart_world",
        "ee_goal_cart_local",
        "ee_goal_orn_quat_world",
        "robot_body_state_w",
        "gripper_actuator_target",
        "gripper_contact_hold_active",
        "basket_payload_normal_reaction_w",
        "basket_payload_friction_reaction_w",
    )

    def __init__(self, env, *, debug=False):
        from gr00t.rl.data_collection.auto_open_standard import B2_JOINT_NAMES, CONTROL_DT

        if abs(float(env.dt) - CONTROL_DT) > 1e-8:
            raise ValueError("Pick collection requires the shared 50 Hz control rate")
        self.env = env
        self.debug = debug
        self.arm_ids = env.arm_joint_ids
        self.b2_ids = torch.tensor(
            [list(env.dof_names).index(name) for name in B2_JOINT_NAMES], device=env.device
        )
        self.gripper_id = int(env.gripper_joint_ids[0])
        self.previous_arm = None
        self._packed = None
        (
            self.states,
            self.actions,
            self.pre,
            self.post,
            self.stages,
            self.sim_times,
            self.contacts,
        ) = [], [], [], [], [], [], []

    def append(self, before, after, stage, *, joint_mode, contacts=()):
        from dataclasses import asdict

        self._packed = None
        state, action = pack_control_frame(
            before,
            after,
            arm_ids=self.arm_ids,
            b2_ids=self.b2_ids,
            gripper_id=self.gripper_id,
            previous_arm_position=self.previous_arm,
            joint_mode=joint_mode,
            dt=float(self.env.dt),
        )
        self.previous_arm = before["dof_pos"][:, self.arm_ids]
        self.states.append(state)
        self.actions.append(action)
        self.pre.append({key: before[key].detach() for key in self.PRE_KEYS})
        post_keys = self.POST_KEYS + tuple(
            key for key in self.DEBUG_POST_KEYS if self.debug and key in after
        )
        self.post.append({key: after[key].detach() for key in post_keys})
        self.stages.append(stage.detach().clone() if torch.is_tensor(stage) else stage)
        self.sim_times.append(float(before["sim_time"]))
        contacts = contacts.items() if isinstance(contacts, dict) else [(0, contacts)]
        for env_id, row_contacts in contacts:
            for contact in row_contacts:
                self.contacts.append(
                    {
                        "episode_index": env_id,
                        "frame_index": len(self.states) - 1,
                        "sim_time_ns": round(float(after["sim_time"]) * 1e9),
                        **asdict(contact),
                    }
                )

    def prepare_finish(self):
        """Assemble/transfer each batch field once, independent of shard count."""
        if self._packed is not None:
            return self._packed
        if not self.states:
            raise ValueError("No control frames were collected")
        contacts = [[] for _ in range(self.env.num_envs)]
        for contact in self.contacts:
            contacts[contact["episode_index"]].append(contact)
        self._packed = {
            "observation_state": torch.stack(self.states).cpu(),
            "action": torch.stack(self.actions).cpu(),
            "pre": {
                key: torch.stack([frame[key] for frame in self.pre]).cpu() for key in self.PRE_KEYS
            },
            "post": {
                key: torch.stack([frame[key] for frame in self.post]).cpu() for key in self.post[0]
            },
            "stages": torch.stack(self.stages).cpu()
            if torch.is_tensor(self.stages[0])
            else self.stages,
            "arm_joint_ids": self.arm_ids.cpu(),
            "b2_joint_ids": self.b2_ids.cpu(),
            "contacts": contacts,
        }
        return self._packed

    def finish(self, *, env_id=None, frame_count=None):
        if not self.states:
            raise ValueError("No control frames were collected")
        if env_id is None:
            if self.env.num_envs != 1:
                raise ValueError("Batch capture requires explicit env_id when finalizing a shard")
            env_id = 0
        if not 0 <= env_id < self.env.num_envs:
            raise ValueError("Shard environment id is outside the batch")
        count = len(self.states) if frame_count is None else frame_count
        if not 0 < count <= len(self.states):
            raise ValueError("Invalid shard frame count")

        packed = self.prepare_finish()

        def row(values):
            # torch.save serializes backing storage, so a CPU row view would
            # accidentally write the whole batch into every episode shard.
            return values[:count, env_id].clone(memory_format=torch.contiguous_format)

        if torch.is_tensor(self.stages[0]):
            from gr00t.rl.pick.stages import BasketStage

            stages = [
                BasketStage(value).name for value in packed["stages"][:count, env_id].tolist()
            ]
        else:
            stages = self.stages[:count]
        return {
            "observation_state": row(packed["observation_state"]),
            "action": row(packed["action"]),
            "pre": {key: row(value) for key, value in packed["pre"].items()},
            "post": {key: row(value) for key, value in packed["post"].items()},
            "stages": stages,
            "sim_times": self.sim_times[:count],
            "contacts": [
                {**contact, "episode_index": 0}
                for contact in packed["contacts"][env_id]
                if contact["frame_index"] < count
            ],
            "joint_names": list(self.env.dof_names),
            "arm_joint_ids": packed["arm_joint_ids"],
            "b2_joint_ids": packed["b2_joint_ids"],
            "gripper_joint_id": self.gripper_id,
        }


def create_pick_contact_audit(env):
    from gr00t.rl.data_collection.contact_audit import SubstepContactAudit

    return SubstepContactAudit.create(env, max_contact_data_per_env=1024)
