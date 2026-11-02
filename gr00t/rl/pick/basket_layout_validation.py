"""Preflight a small GPU shortlist before resetting physical basket payloads."""

import copy
import math
from types import SimpleNamespace

import torch
from legged_gym.utils.ee_steady_inverse_gpu import _matrix_to_quaternion_wxyz

from gr00t.rl.auto_open.motion_math import _quat_multiply
from gr00t.rl.envs.door.b2z1_reset_geometry import b2z1_gripper_pose_and_jacobian
from gr00t.rl.pick.basket_controller import BasketArmCollisionSamples, basket_required_height
from gr00t.rl.pick.basket_geometry import BasketGeometryBatch, BatchedBasketPlanningGeometry
from gr00t.rl.pick.basket_grasp import plan_basket_grasp_batch
from gr00t.rl.pick.basket_initialization import BasketAdmissionInputs
from gr00t.rl.pick.batched_geometry import BatchedCompositeGeometry, BatchedSceneGeometry
from gr00t.rl.pick.pocket import PICK_OBJECT_CONTACT_OFFSET_M
from gr00t.rl.pick.scene_geometry import SceneGeometry


class BasketLayoutPathValidator:
    """Nominal basket/payload/floor paths; actual scene admission still revalidates."""

    def __init__(
        self,
        asset_rows,
        targets,
        basket,
        joints,
        limits,
        collision,
        *,
        geometry_factory=None,
        admission_inputs=None,
        dt=0.02,
        maximum_layouts=1,
        proposal_seeds=None,
    ):
        if not 1 <= maximum_layouts <= 4:
            raise ValueError("Nominal layout shortlist must contain one to four choices")
        self.count = len(asset_rows)
        self.maximum_layouts = maximum_layouts
        self.device = joints.device
        self.limits, self.basket = limits, basket
        self.dt = dt
        self.joints = joints.clone()
        self.roots = joints.new_zeros(self.count, 13)
        self.roots[:, 2], self.roots[:, 3] = 0.55, 1
        self.rows = torch.arange(self.count, device=self.device)
        slots = max(map(len, asset_rows))
        order = [
            [target, *[i for i in range(len(row)) if i != target]]
            for row, target in zip(asset_rows, targets, strict=True)
        ]
        self.order = torch.tensor(
            [row + [0] * (slots - len(row)) for row in order], device=self.device
        )
        episodes = [
            {
                "asset": row[target],
                "distractors": [{"asset": row[i]} for i in range(len(row)) if i != target],
            }
            for row, target in zip(asset_rows, targets, strict=True)
        ]
        ground = SceneGeometry(
            joints.new_tensor([[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]]),
            torch.tensor([[0, 1, 2], [0, 2, 3]], device=self.device),
            self.device,
        )
        if geometry_factory is None:
            self.factory = BasketGeometryBatch(
                [
                    SimpleNamespace(static_scene=ground, basket=basket, episode=episode)
                    for episode in episodes
                ]
            )
        else:
            self.factory = copy.copy(geometry_factory)
            self.factory.rooms = BatchedSceneGeometry(
                [ground],
                torch.zeros(self.count, device=self.device, dtype=torch.long),
                self.roots[:, :3] * 0,
            )
            self.factory.props = [view.select(self.rows) for view in geometry_factory.props]
            self.factory.basket = geometry_factory.basket.select(self.rows)
            self.factory.target = None
            self.factory.nonrobot_geometry = BatchedCompositeGeometry(
                (self.factory.rooms, *self.factory.props)
            )
            self.factory.geometry = BatchedBasketPlanningGeometry(
                self.factory.nonrobot_geometry, self.factory.basket, basket
            )
        self.inputs = admission_inputs or BasketAdmissionInputs(episodes, slots, self.device)
        self.proposal_seeds = proposal_seeds
        self.intrinsic_proposals = None
        # Synthetic planning states contain exactly the same six physical arm joints.
        arm = copy.copy(collision)
        arm.arm_joint_ids = torch.arange(6, device=self.device)
        self.collision = BasketArmCollisionSamples(arm)

    def proposal_bank(self, rows=None):
        if self.proposal_seeds is None:
            return None
        if self.intrinsic_proposals is None:
            from gr00t.rl.pick.basket_intrinsic_proposals import cached_intrinsic_proposals

            self.intrinsic_proposals = cached_intrinsic_proposals(self.inputs, self.proposal_seeds)
        return self.intrinsic_proposals if rows is None else self.intrinsic_proposals.select(rows)

    def certify(self, state, rows, geometry, minimum, hint_mask, hint_orientation):
        accepted = torch.zeros(len(rows), device=self.device, dtype=torch.bool)
        selected_orientation = state["ee_orn"].clone()
        hint_checked = torch.zeros_like(accepted)
        hint_passed = torch.zeros_like(accepted)
        cache_checked = torch.zeros_like(accepted)
        cache_passed = torch.zeros_like(accepted)
        cache = (
            self.proposal_bank(rows) if getattr(self, "proposal_seeds", None) is not None else None
        )
        bank = None if cache is None else cache.candidate_bank(state, self.collision, geometry)
        for source_kind in ("cache", "hint", "full"):
            if source_kind == "cache" and cache is None:
                continue
            available = (
                cache.counts > 0
                if source_kind == "cache"
                else hint_mask
                if source_kind == "hint"
                else ~accepted
            )
            local_rows = torch.where(~accepted & available)[0]
            if not len(local_rows):
                continue
            source = rows[local_rows]
            local = {key: value[local_rows] for key, value in state.items()}
            _, valid, detail = plan_basket_grasp_batch(
                local,
                self.inputs.sizes[source],
                self.limits,
                self.collision,
                geometry.select(local_rows),
                self.dt,
                support_points=self.inputs.support[source],
                support_mask=self.inputs.support_mask[source],
                initial_height=local["object_state"][:, 2],
                minimum_height=minimum[local_rows],
                prefer_realized_body=torch.zeros_like(accepted[local_rows]),
                realized_body=torch.ones_like(accepted[local_rows]),
                minimum_realized_grasp_margin_rad=math.radians(2),
                body_heights=local["root_states"][:, 2:3] + self.roots.new_tensor([-0.01, 0, 0.01]),
                **(
                    {
                        "candidate_bank": (
                            bank[0][local_rows],
                            bank[1][local_rows],
                            {name: value[local_rows] for name, value in bank[2].items()},
                            bank[3],
                            bank[4],
                        )
                    }
                    if source_kind == "cache"
                    else {"candidate_orientations": hint_orientation[local_rows, None]}
                    if source_kind == "hint"
                    else {}
                ),
            )
            accepted[local_rows] = valid
            if "entry_orientation_world_wxyz" in detail:
                selected_orientation[local_rows[valid]] = detail["entry_orientation_world_wxyz"][
                    valid
                ]
            if source_kind == "hint":
                hint_checked[local_rows] = True
                hint_passed[local_rows] = valid
            elif source_kind == "cache":
                cache_checked[local_rows] = True
                cache_passed[local_rows] = valid
        return (
            accepted,
            selected_orientation,
            hint_checked,
            hint_passed,
            cache_checked,
            cache_passed,
        )

    def __call__(self, data, preferred, report, rows, joints):
        self.joints[rows] = joints
        p, rotation, _ = b2z1_gripper_pose_and_jacobian(
            self.roots[:, :3], self.roots[:, [4, 5, 6, 3]], self.joints
        )
        state = {
            "root_states": self.roots,
            "dof_pos": self.joints,
            "ee_pos": p,
            "ee_orn": _matrix_to_quaternion_wxyz(rotation),
            "object_state": self.roots.new_zeros(self.count, 13),
            "distractor_states": self.roots.new_zeros(self.count, self.order.shape[1] - 1, 13),
        }
        n, width = preferred.shape
        owners = torch.arange(n, device=self.device)
        slot = torch.arange(data["chosen"].shape[1], device=self.device)[None].expand_as(
            data["chosen"]
        )
        inverse = torch.full_like(slot, slot.shape[1])
        inverse.scatter_reduce_(
            1, data["chosen"], slot.masked_fill(~data["mask"], slot.shape[1]), reduce="amin"
        )
        source_slot = inverse.clamp_max(slot.shape[1] - 1).gather(1, self.order[rows])
        shortlisted = data["valid"]
        # Box/seed proposals rank layouts; only the original path check can certify them.
        priority = (preferred.long() + shortlisted.long()).masked_fill(~shortlisted, -1)
        ranked = priority.argsort(dim=-1, descending=True, stable=True)
        found = torch.zeros(n, device=self.device, dtype=torch.bool)
        checked = torch.zeros(n, device=self.device, dtype=torch.long)
        hint_checked = torch.zeros_like(checked)
        hint_passed = torch.zeros_like(checked)
        cache_checked = torch.zeros_like(checked)
        cache_passed = torch.zeros_like(checked)
        certified = torch.zeros_like(preferred)
        for cohort in range(min(self.maximum_layouts, width)):
            column = ranked[:, cohort]
            eligible = ~found & shortlisted[owners, column]
            local_rows = torch.where(eligible)[0]
            if not len(local_rows):
                break
            print(
                f"[task:layout] cohort={cohort + 1}/{self.maximum_layouts} rows={len(local_rows)}"
                " nominal_path_preflight",
                flush=True,
            )
            global_rows = rows[local_rows]
            selected = column[local_rows]
            # The spawn gap disappears while settling. Preview the lower rest
            # envelope, not a target still floating above the basket floor.
            positions = torch.cat(
                (
                    data["xy"][local_rows, selected],
                    (
                        data["height"][local_rows, selected]
                        + data["sizes"][local_rows, :, 2] / 2
                        - PICK_OBJECT_CONTACT_OFFSET_M
                    )[..., None],
                ),
                -1,
            ).gather(1, source_slot[local_rows, :, None].expand(-1, -1, 3))
            positions += self.roots[global_rows, None, :3]
            yaw = data["yaw"][local_rows, selected].gather(1, source_slot[local_rows]) / 2
            poses = self.roots.new_zeros(len(local_rows), self.order.shape[1], 13)
            poses[..., :3] = positions
            poses[..., 3], poses[..., 6] = yaw.cos(), yaw.sin()
            state["object_state"][global_rows] = poses[:, 0]
            state["distractor_states"][global_rows] = poses[:, 1:]
            geometry = self.factory.update(state).select(global_rows)
            local = {key: value[global_rows] for key, value in state.items()}
            minimum = basket_required_height(
                self.basket,
                local,
                self.inputs.support[global_rows],
                torch.as_tensor(data.get("lift_margin", 0.1), device=self.device).expand(n)[
                    local_rows
                ],
                support_mask=self.inputs.support_mask[global_rows],
            )
            accepted, orientation, tried_hint, passed_hint, tried_cache, passed_cache = (
                self.certify(
                    local,
                    global_rows,
                    geometry,
                    minimum.expand(len(global_rows)),
                    preferred[local_rows, selected],
                    _quat_multiply(
                        local["root_states"][:, 3:7],
                        report["orientation_local_wxyz"][local_rows, selected],
                    ),
                )
            )
            hint_checked[local_rows] += tried_hint.long()
            hint_passed[local_rows] += passed_hint.long()
            cache_checked[local_rows] += tried_cache.long()
            cache_passed[local_rows] += passed_cache.long()
            checked[local_rows] += 1
            found[local_rows] |= accepted
            certified[local_rows[accepted], selected[accepted]] = True
            report["orientation_local_wxyz"][local_rows[accepted], selected[accepted]] = (
                orientation[accepted]
            )
        report["full_path_layouts_evaluated"] = checked
        report["full_path_layouts_passed"] = certified.sum(-1)
        report["full_path_hint_layouts_evaluated"] = hint_checked
        report["full_path_hint_layouts_passed"] = hint_passed
        report["full_path_cached_contact_layouts_evaluated"] = cache_checked
        report["full_path_cached_contact_layouts_passed"] = cache_passed
        return certified, report
