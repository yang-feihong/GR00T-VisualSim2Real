"""Seeded basket layouts from the deployed robot's actual collision geometry."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.pick.prepare import NoValidInitialScene, reject_preparation
from gr00t.rl.pick.scene_geometry import SceneGeometry
from gr00t.rl.utils.b2z1_asset import ROBOT_USD


@dataclass(frozen=True)
class BasketConfig:
    min_objects: int = 1
    max_objects: int = 4
    object_gap_m: float = 0.006
    wall_gap_m: float = 0.006
    spawn_gap_m: float = 0.75
    arm_joint_noise_rad: float = 0.05
    placement_candidates: int = 1024
    base_candidates: int = 4096
    lift_above_rim_m: float = 0.10


BASKET_GRASP_CLEARANCE_M = 0.006


class NoValidBasketLayout(NoValidInitialScene):
    pass


class BasketGeometry:
    def __init__(self, boxes, names, *, source, floor_static_friction=0.5):
        """Eight true cube corners per collision, expressed in base_link."""
        self.boxes = boxes
        self.names = names
        self.source = str(source)
        self.floor_static_friction = float(floor_static_friction)
        if len(boxes) != len(names) or boxes.shape[1:] != (8, 3):
            raise ValueError("Basket collision boxes must be named eight-corner cubes")
        if "back_box_062" not in names:
            raise ValueError("Deployed basket has no named floor collider back_box_062")
        floor = boxes[names.index("back_box_062")]
        self.floor_lower = floor.amin(0)
        self.floor_upper = floor.amax(0)
        basket = torch.tensor([name.startswith("back_box_") for name in names], device=boxes.device)
        if not bool(basket.any()):
            raise ValueError("No basket wall colliders in deployed asset")
        self.rim_z = boxes[basket, :, 2].amax()
        faces = boxes.new_tensor(
            [
                [0, 2, 3],
                [0, 3, 1],
                [4, 5, 7],
                [4, 7, 6],
                [0, 1, 5],
                [0, 5, 4],
                [2, 6, 7],
                [2, 7, 3],
                [0, 4, 6],
                [0, 6, 2],
                [1, 3, 7],
                [1, 7, 5],
            ],
            dtype=torch.long,
        )
        faces = faces[:, [0, 2, 1]]
        self.faces = (
            faces[None] + torch.arange(len(boxes), device=boxes.device)[:, None, None] * 8
        ).reshape(-1, 3)
        self.geometry = SceneGeometry(boxes.reshape(-1, 3), self.faces, device=str(boxes.device))

    @classmethod
    def from_robot_usd(cls, usd_path=ROBOT_USD, *, device="cuda:0"):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

        stage = Usd.Stage.Open(str(usd_path))
        if stage is None:
            raise ValueError(f"Cannot open deployed robot: {usd_path}")
        bodies = [p for p in stage.Traverse() if p.GetName() == "base_link"]
        if len(bodies) != 1:
            raise ValueError("Expected one base_link in deployed robot")
        body = bodies[0]
        cache = UsdGeom.XformCache()
        inverse = cache.GetLocalToWorldTransform(body).GetInverse()
        names, corners = [], []
        floor_static_friction = 0.5
        for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if not prim.IsA(UsdGeom.Cube):
                raise ValueError(f"Unsupported base collider for basket layout: {prim.GetPath()}")
            half = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2
            matrix = cache.GetLocalToWorldTransform(prim) * inverse
            corners.append(
                [
                    tuple(matrix.Transform(Gf.Vec3d(x, y, z)))
                    for x in (-half, half)
                    for y in (-half, half)
                    for z in (-half, half)
                ]
            )
            names.append(prim.GetName())
            if prim.GetName() == "back_box_062":
                material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
                if material:
                    floor_static_friction = (
                        UsdPhysics.MaterialAPI(material).GetStaticFrictionAttr().Get()
                    )
        return cls(
            torch.tensor(corners, device=device, dtype=torch.float32),
            names,
            source=usd_path,
            floor_static_friction=floor_static_friction,
        )

    def support(self, xy):
        origins = torch.cat((xy, torch.ones_like(xy[..., :1]) * (self.rim_z + 0.1)), -1)
        ends = origins.clone()
        ends[..., 2] = self.floor_lower[2] - 0.1
        hits, normals, _ = self.geometry.rays(origins, ends)
        return hits[..., 2], normals[..., 2].abs()

    def world_mesh(self, root):
        points = self.boxes.reshape(-1, 3)
        world = _quat_rotate(root[..., None, 3:7], points) + root[..., None, :3]
        return world, self.faces


def sample_basket_layout(assets, basket, seed, config=BasketConfig(), *, count=None):
    """Single-episode tools use the same batched spatial placement implementation."""
    from gr00t.rl.pick.basket_layout import sample_layouts

    result = sample_layouts([assets], basket, [seed], config, counts=[count])[0]
    if isinstance(result, NoValidBasketLayout):
        raise result
    return result


def prepare_basket_episode(
    seed,
    scenes_root,
    output,
    *,
    scene_id=None,
    count=None,
    config=BasketConfig(),
    asset_name=None,
    scene_cache=None,
):
    """Single-seed tooling delegates to the same rank-batched preparation."""
    from gr00t.rl.pick.basket_preparation import prepare_basket_batch
    from gr00t.rl.pick.scene_selection import PreparationSceneCache

    result = prepare_basket_batch(
        [seed],
        scenes_root,
        scene_cache or PreparationSceneCache(),
        scene_id=scene_id,
        count=count,
        config=config,
        asset_name=asset_name,
    )[seed]
    if isinstance(result, NoValidInitialScene):
        raise result
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scenes-root", type=Path, default=Path("/datasets/vlnverse_scenes"))
    parser.add_argument("--scene-id")
    parser.add_argument("--count", type=int)
    parser.add_argument("--asset", help="Optional admitted scan instance for a one-category basket")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        prepare_basket_episode(
            args.seed,
            args.scenes_root,
            args.output,
            scene_id=args.scene_id,
            count=args.count,
            asset_name=args.asset,
        )
    except NoValidInitialScene as error:
        reject_preparation(args.output, error)


if __name__ == "__main__":
    main()
