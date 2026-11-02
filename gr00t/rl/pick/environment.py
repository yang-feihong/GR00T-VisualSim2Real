"""Pick-specific scene/rigid-object integration; door environments are unchanged."""

import math

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from legged_gym.envs.manip_loco.manip_loco import ManipLocoIsaacLab
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

from gr00t.rl.pick.planning import minimum_grip_normal_force
from gr00t.rl.pick.pocket import (
    GRIP_DYNAMIC_FRICTION,
    GRIP_STATIC_FRICTION,
    PICK_OBJECT_CONTACT_OFFSET_M,
)


def _spawn_object(descriptor, prim_name):
    return RigidObject(
        RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/{prim_name}",
            spawn=sim_utils.MultiUsdFileCfg(
                usd_path=[item["asset"]["usd_path"] for item in descriptor],
                random_choice=False,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=2,
                    max_depenetration_velocity=1.0,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=PICK_OBJECT_CONTACT_OFFSET_M, rest_offset=0.0
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
        )
    )


class PickEnvironment(ManipLocoIsaacLab):
    # Later heterogeneous USD references must not inherit env0's asset opinions.
    copy_environment_source = True
    reset_policy_history_on_reset = True

    def _configure_canonical_doorman_env_origins(self):
        from legged_gym.utils.episode_rng import canonical_env_origins_for_seeds

        from gr00t.rl.task_runtime import place_environment_origins

        origins = canonical_env_origins_for_seeds(
            [item["seed"] for item in self.pick_episode],
            self.scene.cfg.env_spacing,
            device=self.device,
        )
        place_environment_origins(self.scene, origins)

    def _setup_imported_scene(self):
        pass

    def _update_rgb_camera_world_poses(self, rig, env_id=0):
        from gr00t.rl.pick.cameras import DIAGNOSTIC_VIEWS, diagnostic_pose

        name = next((key for key, value in self._rgb_camera_rigs.items() if value is rig), None)
        if name not in DIAGNOSTIC_VIEWS or not hasattr(self, "pick_object"):
            return super()._update_rgb_camera_world_poses(rig, env_id)
        position, orientation = diagnostic_pose(
            name, self.robot.data.root_state_w[env_id], self.pick_object.data.root_state_w[env_id]
        )
        positions, orientations = self._fill_rgb_camera_pose_arrays(
            rig, position.detach().cpu().numpy(), orientation.detach().cpu().numpy()
        )
        self._set_rgb_camera_world_poses_from_arrays(rig, positions, orientations)

    def __init__(self, cfg, episode):
        self._static_room_replication = None
        self.pick_episode = episode
        if not isinstance(episode, list) or not episode:
            raise ValueError("Pick scenes require a nonempty list of environment descriptors")
        episodes = episode
        forces = [
            minimum_grip_normal_force(
                asset["size_m"],
                asset["mass_kg"],
                aperture_angle_rad=asset.get("grip_aperture_bound_rad"),
            )
            for asset in (item["asset"] for item in episodes)
        ]
        self.pick_min_grip_force_n = torch.tensor(
            forces, device=cfg.sim.device, dtype=torch.float32
        )[:, None]
        self.gripper_control_options = {
            "contact_force_threshold": self.pick_min_grip_force_n,
            "hold_contact_force_threshold": self.pick_min_grip_force_n,
            "free_close_step": 0.024,
            "free_command_lead": 0.060,
            "contact_close_step": 0.006,
            "contact_command_lead": 0.018,
            "hold_command_lead": 0.006,
            "retain_contact_preload": True,
            "maintain_contact_preload": True,
            "bound_contact_preload": True,
            "prevent_reopening_while_closing": False,
        }
        super().__init__(cfg)
        self._arm_raw_pos_targets_for_step = None

    def _setup_scene(self):
        super()._setup_scene()
        stage = sim_utils.get_current_stage()
        from gr00t.rl.pick.cameras import hide_auxiliary_ground_mesh

        hide_auxiliary_ground_mesh(stage, self.cfg.terrain.prim_path)
        episode = self.pick_episode
        scenes = sim_utils.MultiUsdFileCfg(
            usd_path=[item["scene_usd_path"] for item in episode],
            random_choice=False,
        )
        scenes.func("/World/envs/env_.*/VlnverseScene", scenes)
        from gr00t.rl.pick.scene_selection import disable_embedded_physics_scenes

        disable_embedded_physics_scenes(
            stage,
            [
                (f"/World/envs/env_{row}/VlnverseScene", item["scene_usd_path"])
                for row, item in enumerate(episode)
            ],
        )
        from gr00t.rl.pick.scene_replication import register_static_room_replication

        self._static_room_replication = register_static_room_replication(
            stage,
            [
                (f"/World/envs/env_{row}/VlnverseScene", item["scene_usd_path"])
                for row, item in enumerate(episode)
            ],
        )
        self.pick_object = _spawn_object(episode, "PickObject")
        self.scene.rigid_objects["pick_object"] = self.pick_object
        self.pick_distractors = []
        descriptors = []
        for slot in range(max(len(item.get("distractors", [])) for item in episode)):
            descriptors.append(
                [
                    item["distractors"][slot]
                    if slot < len(item.get("distractors", []))
                    else {**item, "object_position": [0, 0, -100.0 - slot], "object_yaw": 0.0}
                    for item in episode
                ]
            )
        for i, descriptor in enumerate(descriptors):
            obj = _spawn_object(descriptor, f"PickDistractor_{i}")
            self.scene.rigid_objects[f"pick_distractor_{i}"] = obj
            self.pick_distractors.append(obj)
            for row, item in enumerate(episode):
                if i < len(item.get("distractors", [])):
                    continue
                root = stage.GetPrimAtPath(f"/World/envs/env_{row}/PickDistractor_{i}")
                UsdGeom.Imageable(root).MakeInvisible()
                for prim in Usd.PrimRange(root):
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
                    if prim.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        material = UsdShade.Material.Define(stage, "/World/PickGripMaterial")
        physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics.CreateStaticFrictionAttr(GRIP_STATIC_FRICTION)
        physics.CreateDynamicFrictionAttr(GRIP_DYNAMIC_FRICTION)
        physics.CreateRestitutionAttr(0.0)
        for env_id in range(len(episode)):
            for suffix in ("Robot/link06", "Robot/gripperMover", "PickObject"):
                body = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/{suffix}")
                UsdShade.MaterialBindingAPI.Apply(body).Bind(
                    material, UsdShade.Tokens.strongerThanDescendants, "physics"
                )
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        self.pick_contact_sensors = []
        for name in ("link06", "gripperMover"):
            sensor = ContactSensor(
                ContactSensorCfg(
                    prim_path=f"/World/envs/env_.*/Robot/{name}",
                    update_period=0.0,
                    # Current-only consumers need no eager substep history refreshes.
                    history_length=0,
                    filter_prim_paths_expr=["/World/envs/env_.*/PickObject"],
                    track_contact_points=True,
                    track_friction_forces=True,
                    max_contact_data_count_per_prim=128,
                )
            )
            self.scene.sensors[f"pick_contact_{name}"] = sensor
            self.pick_contact_sensors.append(sensor)
        self.basket_support_sensors = []
        if episode[0].get("task_type") == "pick_basket":
            for name in (
                "PickObject",
                *[f"PickDistractor_{i}" for i in range(len(self.pick_distractors))],
            ):
                sensor = ContactSensor(
                    ContactSensorCfg(
                        prim_path=f"/World/envs/env_.*/{name}",
                        update_period=0.0,
                        history_length=0,
                        filter_prim_paths_expr=["/World/envs/env_.*/Robot/base_link"],
                        track_friction_forces=True,
                        max_contact_data_count_per_prim=128,
                    )
                )
                self.scene.sensors[f"basket_support_{name}"] = sensor
                self.basket_support_sensors.append(sensor)

    def close(self):
        replication = getattr(self, "_static_room_replication", None)
        if replication is not None:
            replication.close()
            self._static_room_replication = None
        super().close()

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_batch(env_ids)

    def _reset_batch(self, env_ids):
        from gr00t.rl.pick.scene_adapter import batch_reset_tensors

        descriptors = [
            [
                item["distractors"][slot]
                if slot < len(item.get("distractors", []))
                else {**item, "object_position": [0, 0, -100.0 - slot], "object_yaw": 0.0}
                for item in self.pick_episode
            ]
            for slot in range(len(self.pick_distractors))
        ]
        root, joints, objects = batch_reset_tensors(
            self.pick_episode,
            self.scene.env_origins,
            self.default_dof_pos,
            self.arm_joint_ids,
            self.gripper_joint_ids,
            descriptors,
        )
        self.robot.write_root_state_to_sim(root[env_ids], env_ids=env_ids)
        self.robot.write_joint_state_to_sim(
            joints[env_ids], torch.zeros_like(joints[env_ids]), env_ids=env_ids
        )
        self.robot.set_joint_position_target(joints[env_ids], env_ids=env_ids)
        self._joint_position_target_buf[env_ids] = joints[env_ids]
        if hasattr(self, "pick_object"):
            self.pick_object.write_root_state_to_sim(objects[0][env_ids], env_ids=env_ids)
            for obj, values in zip(self.pick_distractors, objects[1:], strict=True):
                obj.write_root_state_to_sim(values[env_ids], env_ids=env_ids)
        if hasattr(self, "teleop_arm_joint_pos_targets"):
            arm = joints[env_ids][:, self.arm_joint_ids]
            # The episode preset replaces the parent's generic reset pose.
            self.teleop_arm_joint_pos_targets[env_ids] = arm
            self.teleop_arm_joint_limited_targets[env_ids] = arm
            self.teleop_cartesian_transition_targets[env_ids] = arm
            self.teleop_arm_joint_control_mask[env_ids] = True
            self.gripper_pos_targets[env_ids] = -math.pi / 3
            self.gripper_actuator_targets[env_ids] = self.gripper_pos_targets[env_ids]

    def reset_payload_rows(self, env_ids):
        from gr00t.rl.pick.scene_adapter import basket_payload_reset_tensors

        descriptors = [self.pick_episode] + [
            [
                item["distractors"][slot]
                if slot < len(item.get("distractors", []))
                else {"object_position": [0, 0, -100.0 - slot]}
                for item in self.pick_episode
            ]
            for slot in range(len(self.pick_distractors))
        ]
        states = basket_payload_reset_tensors(
            descriptors, self.robot.data.root_state_w, self.scene.env_origins
        )
        for obj, values in zip((self.pick_object, *self.pick_distractors), states, strict=True):
            obj.write_root_state_to_sim(values[env_ids], env_ids=env_ids)

    def get_control_state(self):
        state = super().get_control_state()
        state["object_state"] = self.pick_object.data.root_state_w.clone()
        state["distractor_states"] = (
            torch.stack([obj.data.root_state_w for obj in self.pick_distractors], dim=1).clone()
            if self.pick_distractors
            else state["object_state"].new_empty(self.num_envs, 0, 13)
        )
        forces = [sensor.data.force_matrix_w[:, 0, 0] for sensor in self.pick_contact_sensors]
        state["object_finger_forces"] = torch.stack(forces, 1)
        state["object_finger_friction_forces"] = torch.stack(
            [sensor.data.friction_forces_w[:, 0, 0] for sensor in self.pick_contact_sensors], 1
        )
        state["object_finger_contact_points"] = torch.stack(
            [sensor.data.contact_pos_w[:, 0, 0] for sensor in self.pick_contact_sensors], 1
        )
        state["gripper_actuator_target"] = self.gripper_actuator_targets.clone()
        state["gripper_contact_hold_active"] = self.gripper_contact_hold_active.clone()
        if self.basket_support_sensors:
            # Match ContactSensor.net_forces_w: normal forces exclude friction.
            state["basket_payload_normal_reaction_w"] = -torch.stack(
                [sensor.data.force_matrix_w[:, 0, 0] for sensor in self.basket_support_sensors]
            ).sum(0)
            state["basket_payload_friction_reaction_w"] = -torch.stack(
                [sensor.data.friction_forces_w[:, 0, 0] for sensor in self.basket_support_sensors]
            ).sum(0)
        return state

    def get_gripper_target_contact_sensors(self):
        return self.pick_contact_sensors
