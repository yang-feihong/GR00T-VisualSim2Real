# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Callable, Literal, Optional
import os
import numpy as np
from torch.backends.cudnn import deterministic

import omni.usd
from pxr import Usd, UsdGeom, Gf, Sdf, UsdPhysics
import omni.kit.commands

import isaacsim.core.utils.prims as prim_utils

from isaaclab.sim.utils import bind_visual_material, clone
import isaaclab.sim as sim_utils
from isaaclab.sim import schemas
from isaaclab.utils import configclass
from gr00t.rl.isaac_utils.playground.utils.math_utils import set_prim_transform, Rotation
from gr00t.rl.isaac_utils.playground.utils.usd_utils import add_rigid_body, add_mass, add_collider, create_prim, preload_materials, write_custom_data_to_prim, get_custom_data_from_prim


@configclass
class DeterministicTableCfg:
    translation_x_offset: float = 0.0
    translation_y_offset: float = 0.0
    yaw_deg_offset: float = 0.0
    table_width: float = 1.5
    table_depth: float = 0.8
    table_thickness: float = 0.03
    table_height: float = 0.7
    leg_type: Literal["cube", "cylinder", "hang"] = "cylinder"
    leg_width: float = 0.05
    leg_to_edge_distance: float = 0.1


@configclass
class TableGridSpawnerCfg(sim_utils.RigidObjectSpawnerCfg):
    num_tables: tuple[int, int] = (3, 3)  # number of tables along x and y
    table_spacing: tuple[float, float] = (2.0, 2.0)  # spacing between tables along x and y
    table_width: tuple[float, float] = (1.2, 1.5)  # min and max width
    table_depth: tuple[float, float] = (0.4, 0.6)  # min and max depth
    table_thickness: tuple[float, float] = (0.01, 0.02)  # min and max thickness
    table_height: tuple[float, float] = (0.65, 0.75)  # min and max height
    translation_x_offset_range: tuple[float, float] = (-0.1, 0.1)  # min and max x offset from grid
    translation_y_offset_range: tuple[float, float] = (-0.1, 0.1)  # min and max y offset from grid
    yaw_deg_offset_range: tuple[float, float] = (-5.0, 5.0)  # min and max yaw offset (degree)
    table_top_collider: bool = True
    kinematic: bool = True
    spawn_leg: bool = True
    leg_collider: bool = True
    leg_type: list[Literal["cube", "cylinder", "hang"]] = ["cube", "cylinder", "hang"]
    leg_width: tuple[float, float] = (0.02, 0.04)  # min and max leg width / diameter
    leg_to_edge_distance: tuple[float, float] = (0.03, 0.09)  # min and max distance from edge to leg
    randomize_material: bool = False
    table_top_material_prim_paths: list[str] = []
    leg_material_prim_paths: list[str] = []
    use_preloaded_materials: bool = False
    preloaded_materials_num_transform: int = 1
    preloaded_materials_num_color: int = 1

    # deterministic generation
    deterministic_cfg: Optional[list[list[DeterministicTableCfg]]] = None


@clone
def spawn_table_grid(
    prim_path: str,
    cfg: TableGridSpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    stage: Usd.Stage = omni.usd.get_context().get_stage()

    # gather the configurations
    num_tables_x, num_tables_y = cfg.num_tables
    table_spacing_x, table_spacing_y = cfg.table_spacing

    if cfg.use_preloaded_materials:
        table_top_material_prim_paths, leg_material_prim_paths = preload_table_grid_materials(stage, cfg.preloaded_materials_num_transform, cfg.preloaded_materials_num_color)
    else:
        table_top_material_prim_paths = cfg.table_top_material_prim_paths
        leg_material_prim_paths = cfg.leg_material_prim_paths

    # define the single-table creation function
    def spawn_table(table_prim_path: str, table_x_idx: int, table_y_idx: int) -> DeterministicTableCfg:
        create_prim(table_prim_path, "Xform")

        if cfg.deterministic_cfg is None:
            translation_x_offset = np.random.uniform(cfg.translation_x_offset_range[0], cfg.translation_x_offset_range[1])
            translation_y_offset = np.random.uniform(cfg.translation_y_offset_range[0], cfg.translation_y_offset_range[1])
            yaw_offset = np.random.uniform(cfg.yaw_deg_offset_range[0], cfg.yaw_deg_offset_range[1])
            table_width = np.random.uniform(cfg.table_width[0], cfg.table_width[1])
            table_depth = np.random.uniform(cfg.table_depth[0], cfg.table_depth[1])
            table_thickness = np.random.uniform(cfg.table_thickness[0], cfg.table_thickness[1])
            table_height = np.random.uniform(cfg.table_height[0], cfg.table_height[1])
        else:
            translation_x_offset = cfg.deterministic_cfg[table_x_idx][table_y_idx].translation_x_offset
            translation_y_offset = cfg.deterministic_cfg[table_x_idx][table_y_idx].translation_y_offset
            yaw_offset = cfg.deterministic_cfg[table_x_idx][table_y_idx].yaw_deg_offset
            table_width = cfg.deterministic_cfg[table_x_idx][table_y_idx].table_width
            table_depth = cfg.deterministic_cfg[table_x_idx][table_y_idx].table_depth
            table_thickness = cfg.deterministic_cfg[table_x_idx][table_y_idx].table_thickness
            table_height = cfg.deterministic_cfg[table_x_idx][table_y_idx].table_height

        # set the table transform
        trans_x = (table_spacing_x * table_x_idx) - (table_spacing_x * (num_tables_x - 1) / 2) + translation_x_offset
        trans_y = (table_spacing_y * table_y_idx) - (table_spacing_y * (num_tables_y - 1) / 2) + translation_y_offset
        trans_z = table_height
        set_prim_transform(
            stage, table_prim_path,
            translation=(trans_x, trans_y, trans_z),
            rotation=(0.0, 0.0, yaw_offset),
            scale=(1.0, 1.0, 1.0)
        )

        # spawn the table top
        table_top_prim_path = os.path.join(table_prim_path, "table_top")
        create_prim(table_top_prim_path, "Cube")
        set_prim_transform(
            stage, table_top_prim_path,
            translation=(0.0, 0.0, -table_thickness / 2),
            rotation=(0.0, 0.0, 0.0),
            scale=(table_depth / 2, table_width / 2, table_thickness / 2)
        )
        if cfg.table_top_collider:
            add_collider(stage, table_top_prim_path)
        if cfg.randomize_material:
            table_top_material_prim_path = np.random.choice(table_top_material_prim_paths)
            bind_visual_material(table_top_prim_path, table_top_material_prim_path, stage)

        # spawn the legs
        if cfg.spawn_leg:
            if cfg.deterministic_cfg is None:
                leg_type = np.random.choice(cfg.leg_type)
                leg_width = np.random.uniform(cfg.leg_width[0], cfg.leg_width[1])
                leg_to_edge_distance = np.random.uniform(cfg.leg_to_edge_distance[0], cfg.leg_to_edge_distance[1])
            else:
                leg_type = cfg.deterministic_cfg[table_x_idx][table_y_idx].leg_type
                leg_width = cfg.deterministic_cfg[table_x_idx][table_y_idx].leg_width
                leg_to_edge_distance = cfg.deterministic_cfg[table_x_idx][table_y_idx].leg_to_edge_distance
            if cfg.randomize_material:
                leg_material_prim_path = np.random.choice(leg_material_prim_paths)
            if leg_type in ["cube", "cylinder"]:
                leg_type = leg_type.capitalize()
                leg_translations = [
                    (-table_depth / 2 + leg_to_edge_distance, -table_width / 2 + leg_to_edge_distance, -table_height / 2 - table_thickness / 2),
                    (-table_depth / 2 + leg_to_edge_distance, table_width / 2 - leg_to_edge_distance, -table_height / 2 - table_thickness / 2),
                    (table_depth / 2 - leg_to_edge_distance, -table_width / 2 + leg_to_edge_distance, -table_height / 2 - table_thickness / 2),
                    (table_depth / 2 - leg_to_edge_distance, table_width / 2 - leg_to_edge_distance, -table_height / 2 - table_thickness / 2),
                ]

                for i, leg_translation in enumerate(leg_translations):
                    leg_prim_path = os.path.join(table_prim_path, f"leg_{i}")
                    create_prim(leg_prim_path, leg_type)
                    if leg_type == "Cube":
                        scale = (leg_width / 2, leg_width / 2, table_height / 2 - table_thickness / 2)
                    elif leg_type == "Cylinder":
                        scale = (1.0, 1.0, 1.0)
                    set_prim_transform(
                        stage, leg_prim_path,
                        translation=leg_translation,
                        rotation=(0.0, 0.0, 0.0),
                        scale=scale
                    )
                    if leg_type == "Cylinder":
                        leg_geom: UsdGeom.Cylinder = UsdGeom.Cylinder.Define(stage, leg_prim_path)
                        leg_geom.GetRadiusAttr().Set(leg_width / 2)
                        leg_geom.GetHeightAttr().Set(table_height - table_thickness)
                    if cfg.leg_collider:
                        add_collider(stage, leg_prim_path)
                    if cfg.randomize_material:
                        bind_visual_material(leg_prim_path, leg_material_prim_path, stage)
            elif leg_type == "hang":
                left_support_beam_prim_path = os.path.join(table_prim_path, "left_support_beam")
                create_prim(left_support_beam_prim_path, "Cube")
                set_prim_transform(
                    stage, left_support_beam_prim_path,
                    translation=(0.0, table_width / 2 - leg_to_edge_distance, -leg_width / 2 - table_thickness),
                    rotation=(0.0, 0.0, 0.0),
                    scale=(table_depth / 2 - leg_to_edge_distance / 2, leg_width / 2, leg_width / 2)
                )

                right_support_beam_prim_path = os.path.join(table_prim_path, "right_support_beam")
                create_prim(right_support_beam_prim_path, "Cube")
                set_prim_transform(
                    stage, right_support_beam_prim_path,
                    translation=(0.0, -table_width / 2 + leg_to_edge_distance, -leg_width / 2 - table_thickness),
                    rotation=(0.0, 0.0, 0.0),
                    scale=(table_depth / 2 - leg_to_edge_distance / 2, leg_width / 2, leg_width / 2)
                )

                left_leg_prim_path = os.path.join(table_prim_path, "left_leg")
                create_prim(left_leg_prim_path, "Cube")
                set_prim_transform(
                    stage, left_leg_prim_path,
                    translation=(0.0, table_width / 2 - leg_to_edge_distance, -table_height / 2 - table_thickness / 2),
                    rotation=(0.0, 0.0, 0.0),
                    scale=(leg_width / 2, leg_width / 2, table_height / 2 - table_thickness / 2 - leg_width)
                )

                right_leg_prim_path = os.path.join(table_prim_path, "right_leg")
                create_prim(right_leg_prim_path, "Cube")
                set_prim_transform(
                    stage, right_leg_prim_path,
                    translation=(0.0, -table_width / 2 + leg_to_edge_distance, -table_height / 2 - table_thickness / 2),
                    rotation=(0.0, 0.0, 0.0),
                    scale=(leg_width / 2, leg_width / 2, table_height / 2 - table_thickness / 2 - leg_width)
                )

                left_bottom_beam_prim_path = os.path.join(table_prim_path, "left_bottom_beam")
                create_prim(left_bottom_beam_prim_path, "Cube")
                set_prim_transform(
                    stage, left_bottom_beam_prim_path,
                    translation=(0.0, table_width / 2 - leg_to_edge_distance, -table_height + leg_width / 2),
                    rotation=(0.0, 0.0, 0.0),
                    scale=(table_depth / 2 - leg_to_edge_distance / 2, leg_width / 2, leg_width / 2)
                )

                right_bottom_beam_prim_path = os.path.join(table_prim_path, "right_bottom_beam")
                create_prim(right_bottom_beam_prim_path, "Cube")
                set_prim_transform(
                    stage, right_bottom_beam_prim_path,
                    translation=(0.0, -table_width / 2 + leg_to_edge_distance, -table_height + leg_width / 2),
                    rotation=(0.0, 0.0, 0.0),
                    scale=(table_depth / 2 - leg_to_edge_distance / 2, leg_width / 2, leg_width / 2)
                )

                if cfg.leg_collider:
                    add_collider(stage, left_support_beam_prim_path)
                    add_collider(stage, right_support_beam_prim_path)
                    add_collider(stage, left_leg_prim_path)
                    add_collider(stage, right_leg_prim_path)
                    add_collider(stage, left_bottom_beam_prim_path)
                    add_collider(stage, right_bottom_beam_prim_path)
                if cfg.randomize_material:
                    bind_visual_material(left_support_beam_prim_path, leg_material_prim_path, stage)
                    bind_visual_material(right_support_beam_prim_path, leg_material_prim_path, stage)
                    bind_visual_material(left_leg_prim_path, leg_material_prim_path, stage)
                    bind_visual_material(right_leg_prim_path, leg_material_prim_path, stage)
                    bind_visual_material(left_bottom_beam_prim_path, leg_material_prim_path, stage)
                    bind_visual_material(right_bottom_beam_prim_path, leg_material_prim_path, stage)

            else:
                raise ValueError(f"Invalid leg type: {leg_type}")

            return DeterministicTableCfg(
                translation_x_offset=translation_x_offset,
                translation_y_offset=translation_y_offset,
                yaw_deg_offset=yaw_offset,
                table_width=table_width,
                table_depth=table_depth,
                table_thickness=table_thickness,
                table_height=table_height,
                leg_type=leg_type.lower(),
                leg_width=leg_width,
                leg_to_edge_distance=leg_to_edge_distance
            )

    # spawn the table grid root prim
    create_prim(prim_path, "Xform")
    if translation is None:
        translation = (0.0, 0.0, 0.0)
    if orientation is None:
        orientation = (1.0, 0.0, 0.0, 0.0)
    set_prim_transform(
        stage, prim_path,
        translation=translation,
        rotation=Rotation(*orientation).to_euler_xyz(degrees=True),
        scale=(1.0, 1.0, 1.0)
    )
    add_rigid_body(stage, prim_path, kinematic=cfg.kinematic)
    add_mass(stage, prim_path, mass=100.0)

    # spawn the table grid
    for table_x_idx in range(num_tables_x):
        for table_y_idx in range(num_tables_y):
            table_prim_path = os.path.join(prim_path, f"table_{table_x_idx}_{table_y_idx}")
            deterministic_table_cfg = spawn_table(table_prim_path, table_x_idx, table_y_idx)
            write_custom_data_to_prim(stage, prim_path, {f"TableX{table_x_idx}Y{table_y_idx}": deterministic_table_cfg.to_dict()})

    return prim_utils.get_prim_at_path(prim_path)


def preload_table_grid_materials(stage: Usd.Stage, random_transform_each_num: int = 1, random_color_each_num: int = 1):
    textured_material_list = [
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Wood/Ash.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Wood/Ash_Planks.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Wood/Bamboo.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Wood/Bamboo_Planks.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Wood/Beadboard.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Wood/Birch.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Wood/Birch_Planks.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Wood/Cherry.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Wood/Cherry_Planks.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Wood/Cork.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Wood/Mahogany.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Wood/Mahogany_Planks.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Wood/Oak.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Wood/Oak_Planks.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Wood/Parquet_Floor.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Wood/Plywood.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Wood/Timber.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Wood/Timber_Cladding.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Wood/Walnut.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Wood/Walnut_Planks.mdl",

        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Plastics/Plastic.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Plastics/Plastic_ABS.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Plastics/Rubber_Smooth.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Plastics/Veneer_OU_Walnut.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Plastics/Veneer_UX_Walnut_Cherry.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Plastics/Veneer_Z5_Maple.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Plastics/Vinyl.mdl",
    ]

    random_color_material_list = [
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Miscellaneous/Paint_Gloss.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Miscellaneous/Paint_Gloss_Finish.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Miscellaneous/Paint_Matte.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Miscellaneous/Paint_Matte_Finish.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Miscellaneous/Paint_Satin.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Miscellaneous/Paint_Satin_Finish.mdl",
    ]

    metallic_material_list = [
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Metals/Aluminum_Anodized.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Metals/Aluminum_Anodized_Black.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Metals/Aluminum_Anodized_Blue.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Metals/Aluminum_Anodized_Charcoal.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Metals/Aluminum_Anodized_Red.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Metals/Aluminum_Cast.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Metals/Aluminum_Polished.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Metals/Brass.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Metals/Bronze.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Metals/Brushed_Antique_Copper.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Metals/Cast_Metal_Silver_Vein.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Metals/Chrome.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Metals/Copper.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Metals/CorrugatedMetal.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Metals/Gold.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Metals/Iron.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Metals/Metal_Door.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Metals/Metal_Seamed_Roof.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Metals/RustedMetal.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Metals/Silver.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Metals/Steel_Blued.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Metals/Steel_Carbon.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Metals/Steel_Cast.mdl",
        "omniverse://nucleus.westus2.cloudapp.azure.com/NVIDIA/Materials/Base/Metals/Steel_Stainless.mdl",
    ]

    table_top_material_prim_paths = preload_materials(stage, textured_material_list, "/World/Looks/TableTopMaterials", random_transform_each_num=random_transform_each_num, random_color_material_list=random_color_material_list, random_color_each_num=random_color_each_num)
    leg_material_prim_paths = preload_materials(stage, metallic_material_list, "/World/Looks/LegMaterials", random_transform_each_num=random_transform_each_num)

    return table_top_material_prim_paths, leg_material_prim_paths


def get_deterministic_table_grid_config(cfg: TableGridSpawnerCfg, metadata: dict) -> TableGridSpawnerCfg:
    cfg.deterministic_cfg = []
    for table_x_idx in range(cfg.num_tables[0]):
        cfg.deterministic_cfg.append([])
        for table_y_idx in range(cfg.num_tables[1]):
            table_cfg = DeterministicTableCfg()
            DeterministicTableCfg.from_dict(obj=table_cfg, data=metadata[f"TableX{table_x_idx}Y{table_y_idx}"])
            cfg.deterministic_cfg[table_x_idx].append(table_cfg)
    return cfg
