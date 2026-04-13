# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import omni.usd
from gr00t.rl.isaac_utils.playground.env_rand.table_grid import (
    TableGridSpawnerCfg,
    get_deterministic_table_grid_config,
    spawn_table_grid,
)
from gr00t.rl.isaac_utils.playground.utils.usd_utils import preload_materials

from gr00t.rl.simulator.isaacsim.isaacsim import DEFAULT_ISAACSIM_VERSION
from gr00t.rl.utils.teleop_reset_lib.motion_loader_wsppt import MotionLoaderWSPPT

if DEFAULT_ISAACSIM_VERSION == "4.5":
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import (
        ActuatorNetMLPCfg,
        DCMotorCfg,
        IdealPDActuatorCfg,
        ImplicitActuatorCfg,
    )
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg


elif DEFAULT_ISAACSIM_VERSION == "4.2":

    import omni.isaac.lab.sim as sim_utils
    from omni.isaac.lab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg
    from omni.isaac.lab.assets.articulation import ArticulationCfg, Articulation
    from omni.isaac.lab.assets.rigid_object import RigidObjectCfg, RigidObject

import os

import numpy as np
from loguru import logger

stage = omni.usd.get_context().get_stage()

textured_material_list = [
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Ash.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Ash_Planks.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Bamboo.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Bamboo_Planks.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Beadboard.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Birch.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Birch_Planks.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Cherry.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Cherry_Planks.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Cork.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Mahogany.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Mahogany_Planks.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Oak.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Oak_Planks.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Parquet_Floor.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Plywood.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Timber.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Timber_Cladding.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Walnut.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Walnut_Planks.mdl",

    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Plastics/Plastic.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Plastics/Plastic_ABS.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Plastics/Rubber_Smooth.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Plastics/Veneer_OU_Walnut.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Plastics/Veneer_UX_Walnut_Cherry.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Plastics/Veneer_Z5_Maple.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Plastics/Vinyl.mdl",
]

random_color_material_list = [
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Miscellaneous/Paint_Gloss.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Miscellaneous/Paint_Gloss_Finish.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Miscellaneous/Paint_Matte.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Miscellaneous/Paint_Matte_Finish.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Miscellaneous/Paint_Satin.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Miscellaneous/Paint_Satin_Finish.mdl",
]

metallic_material_list = [
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Aluminum_Anodized.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Aluminum_Anodized_Black.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Aluminum_Anodized_Blue.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Aluminum_Anodized_Charcoal.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Aluminum_Anodized_Red.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Aluminum_Cast.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Aluminum_Polished.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Brass.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Bronze.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Brushed_Antique_Copper.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Cast_Metal_Silver_Vein.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Chrome.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Copper.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/CorrugatedMetal.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Gold.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Iron.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Metal_Door.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Metal_Seamed_Roof.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/RustedMetal.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Silver.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Steel_Blued.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Steel_Carbon.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Steel_Cast.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Steel_Stainless.mdl",
]

table_top_material_prim_paths = preload_materials(stage, textured_material_list, "/World/Looks/TableTopMaterials", random_transform_each_num=5, random_color_material_list=random_color_material_list, random_color_each_num=5)
leg_material_prim_paths = preload_materials(stage, metallic_material_list, "/World/Looks/LegMaterials", random_transform_each_num=5)
_materials_available = bool(table_top_material_prim_paths) and bool(leg_material_prim_paths)
if not _materials_available:
    print("Warning: No materials loaded from Nucleus. Table material randomization will be disabled.")


# Ensure USD_PATH_LIST is initialized at import time, before any function calls
# Use absolute path relative to this file for USD_DIR
USD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../objects/grab")
# USD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../objects/objaverse")
USD_DIR = os.path.normpath(USD_DIR)
USD_PATH_LIST = []
USD_ASSET_IDX_LIST = []
USD_ASSET_CLASS_NAMES = []  # Store the class names corresponding to each USD asset

def _init_usd_path_list():
    global USD_PATH_LIST
    global USD_ASSET_IDX_LIST
    global USD_ASSET_CLASS_NAMES
    # asset_include_keyname_list = [ 'apple', 'alarmclock', 'coffeemug', 'cup', 'duck', 'doorknob', 'mug', 'waterbottle', 'wineglass', 'bottle']
    asset_include_keyname_list = ['bottle']
    # Tairan: @zi, please make the above hard-coded keyname list to be scalable with objaverse
    # asset_include_keyname_list = ['bottle']

    if not USD_PATH_LIST:  # Only initialize once
        for root, dirs, files in os.walk(USD_DIR):
            # Sort directories and files to ensure consistent ordering
            dirs.sort()
            files.sort()
            for file_name in files:
                # logger.info(f"File name: {file_name}")
                if file_name.endswith(".usda") or file_name.endswith(".usd"):
                    if 'instanceable_meshes' not in file_name:
                        # if any(keyname in file_name for keyname in asset_include_keyname_list):
                        if any(file_name.rsplit('.', 1)[0] == keyname for keyname in asset_include_keyname_list):
                        # if file_name.rsplit('.', 1)[0] == asset_include_keyname_list[0]:
                            # Store relative path from repo root (assume cwd is repo root)
                            rel_path = os.path.relpath(os.path.join(root, file_name), os.getcwd())
                            USD_PATH_LIST.append(rel_path)
                            # Store the class name (filename without extension)
                            class_name = file_name.rsplit('.', 1)[0]
                            USD_ASSET_CLASS_NAMES.append(class_name)

        # Sort the final lists to ensure consistent ordering across runs
        paired_lists = list(zip(USD_PATH_LIST, USD_ASSET_CLASS_NAMES))
        paired_lists.sort(key=lambda x: x[1])  # Sort by class name
        USD_PATH_LIST, USD_ASSET_CLASS_NAMES = zip(*paired_lists) if paired_lists else ([], [])
        USD_PATH_LIST = list(USD_PATH_LIST)
        USD_ASSET_CLASS_NAMES = list(USD_ASSET_CLASS_NAMES)
        logger.info(f"Found {len(USD_PATH_LIST)} USD files in {USD_DIR}")
        logger.info(f"Object classes: {USD_ASSET_CLASS_NAMES}")
        logger.info("USD_PATH_LIST: ", USD_PATH_LIST)

# Initialize USD_PATH_LIST at import time so it's available when imported from other files
_init_usd_path_list()

# def get_table_grid_spawner_cfg():
#     table_grid_spawner_cfg = TableGridSpawnerCfg(
#     func=spawn_table_grid,
#     activate_contact_sensors=True,
#     num_tables=(2, 1),
#     table_depth=(0.7, 0.75),
#     table_width=(1.4, 1.6),
#     table_thickness=(0.035, 0.04),
#     table_height=(0.65, 0.75),
#     # randomize_material=True,
#     # use_preloaded_materials=True,
#     translation_x_offset_range=(-0.1, 0.1),
#     # table_spacing=(4.0, 4.0),
#     table_spacing=(5.4, 5.4), # this is for aligned previous bug-free two tables spacing
#     randomize_material=True,
#     table_top_material_prim_paths=table_top_material_prim_paths,
#     leg_material_prim_paths=leg_material_prim_paths,
# )
#     return table_grid_spawner_cfg

def get_table_grid_spawner_cfg(env_config, env_idx, sim_device):

    # load 
    # table_grid_meta_data = {'TableX0Y0': {'leg_to_edge_distance': 0.04817398602107347, 'leg_type': 'cylinder', 'leg_width': 0.02463449210241813, 'table_depth': 0.6937774101187943, 'table_height': 0.380555544497443, 'table_thickness': 0.0359727289479935, 'table_width': 1.713555265825692, 'translation_x_offset': 0.00020127857021027466, 'translation_y_offset': -0.0004597068936516707, 'yaw_deg_offset': -4.162707481429723},
    #                         'TableX1Y0': {'leg_to_edge_distance': 0.04817398602107347, 'leg_type': 'cylinder', 'leg_width': 0.02463449210241813, 'table_depth': 0.6937774101187943, 'table_height': 0.380555544497443, 'table_thickness': 0.0359727289479935, 'table_width': 1.713555265825692, 'translation_x_offset': 0.00020127857021027466, 'translation_y_offset': -0.0004597068936516707, 'yaw_deg_offset': -4.162707481429723}}
    

    
    table_grid_spawner_cfg = TableGridSpawnerCfg(
    func=spawn_table_grid,
    activate_contact_sensors=True,
    num_tables=(2, 1),
    table_depth=(0.7, 0.75),
    table_width=(1.4, 1.6),
    table_thickness=(0.035, 0.04),
    table_height=(0.674, 0.6741),
    # randomize_material=True,
    # use_preloaded_materials=True,
    translation_x_offset_range=(0.089, 0.09),
    # table_spacing=(4.0, 4.0),
    table_spacing=(5.4, 5.4), # this is for aligned previous bug-free two tables spacing
    randomize_material=_materials_available, # TODO: Need to manually set the table material in the config to be true if you want to randomize the table material
    table_top_material_prim_paths=table_top_material_prim_paths,
    leg_material_prim_paths=leg_material_prim_paths,
    )
    if env_config.reset_from_dataset.enable:
        table_grid_meta_data = get_table_grid_meta_data(env_config, env_idx, sim_device)
        return get_deterministic_table_grid_config(table_grid_spawner_cfg, table_grid_meta_data)
    return table_grid_spawner_cfg

def get_table_grid_meta_data(env_config, env_idx, sim_device):
    motion_dir = env_config.reset_from_dataset.motion_file_dir
    all_motion_files_under_dir = [
        os.path.join(motion_dir, f)
        for f in os.listdir(motion_dir)
        if f.endswith('.npz')
    ]


    motion_idx_for_this_env_idx = env_idx % len(all_motion_files_under_dir)
    # logger.info(f"env_idx: {env_idx}")
    # logger.info(f"len(all_motion_files_under_dir): {len(all_motion_files_under_dir)}")
    # logger.info(f"motion_idx_for_this_env_idx: {motion_idx_for_this_env_idx}")
    # logger.info(f"motion_file: {all_motion_files_under_dir[motion_idx_for_this_env_idx]}")
    motion_loader = MotionLoaderWSPPT.from_file(all_motion_files_under_dir[motion_idx_for_this_env_idx], sim_device)
    table_grid_meta_data = motion_loader.table_custom_data
    return table_grid_meta_data

# def get_table_grid_spawner_cfg():
#     table_grid_spawner_cfg = TableGridSpawnerCfg(
#     func=spawn_table_grid,
#     activate_contact_sensors=True,
#     num_tables=(2, 1),
#     table_spacing=(4.0, 4.0),
#     randomize_material=True,
#     table_top_material_prim_paths=table_top_material_prim_paths,
#     leg_material_prim_paths=leg_material_prim_paths,
# )
#     return table_grid_spawner_cfg

# table_multi_spawner_cfg = sim_utils.MultiAssetSpawnerCfg(
#     assets_cfg=[get_table_grid_spawner_cfg() for _ in range(4096)],  # TODO(haoru): change to num_env if there is way to pass it in
#     random_choice=False,
#     activate_contact_sensors=True,
#     rigid_props=sim_utils.RigidBodyPropertiesCfg(
#         disable_gravity=False,
#         kinematic_enabled=True,
#         retain_accelerations=False,
#         linear_damping=0.0,
#         angular_damping=0.0,
#         max_linear_velocity=1000.0,
#         max_angular_velocity=1000.0,
#         max_depenetration_velocity=1.0,
#     ),
# )

# def get_cylider_spawner_cfg():
#     friction_coef = np.random.uniform(0.2, 0.6)
#     return sim_utils.CylinderCfg(
#         radius=np.random.uniform(0.025, 0.035),
#         height=np.random.uniform(0.14, 0.16),
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
#         visual_material=sim_utils.PreviewSurfaceCfg(),
#         collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
#         physics_material=sim_utils.RigidBodyMaterialCfg(
#             static_friction=friction_coef,
#             dynamic_friction=friction_coef
#         ),
#         mass_props=sim_utils.MassPropertiesCfg(
#             mass=np.random.uniform(0.1, 0.15),
#         )
#     )
# cylinder_multi_spawner_cfg = sim_utils.MultiAssetSpawnerCfg(
#     assets_cfg=[get_cylider_spawner_cfg() for _ in range(4096)],
#     random_choice=False,
#     activate_contact_sensors=True,
#     rigid_props=sim_utils.RigidBodyPropertiesCfg(
#         retain_accelerations=False,
#         linear_damping=0.0,
#         angular_damping=0.0,
#         max_linear_velocity=1000.0,
#         max_angular_velocity=1000.0,
#         max_depenetration_velocity=1.0,
#     ),
# )

def get_random_object_spawner_cfg():
    
    friction_coef = np.random.uniform(0.2, 0.6)
    if not USD_PATH_LIST:
        raise RuntimeError("USD_PATH_LIST is empty. Make sure _init_usd_path_list() was called and USD_DIR is correct.")
    random_usd_idx = np.random.randint(0, len(USD_PATH_LIST))
    random_usd_path = USD_PATH_LIST[random_usd_idx]
    global USD_ASSET_IDX_LIST
    USD_ASSET_IDX_LIST.append(random_usd_idx)
    # print(f"USD_ASSET_IDX_LIST: {len(USD_ASSET_IDX_LIST)}")
    # logger.info(f"random_usd_path: {random_usd_path}")
    return sim_utils.UsdFileCfg(
            activate_contact_sensors=True,
            usd_path=random_usd_path,
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        )

# random_object_multi_spawner_cfg = sim_utils.MultiAssetSpawnerCfg(
#     assets_cfg=[get_random_object_spawner_cfg() for _ in range(4096)],
#     random_choice=False,
#     activate_contact_sensors=True,
#     rigid_props=sim_utils.RigidBodyPropertiesCfg(
#         retain_accelerations=False,
#         linear_damping=0.0,
#         angular_damping=0.0,
#         max_linear_velocity=1000.0,
#         max_angular_velocity=1000.0,
#         max_depenetration_velocity=1.0,
#     ),
# )

def get_table_multi_spawner_cfg_based_on_num_envs(num_envs, env_config, sim_device):
    return sim_utils.MultiAssetSpawnerCfg(
    assets_cfg=[get_table_grid_spawner_cfg(env_config, env_idx, sim_device) for env_idx in range(num_envs)],  # TODO(haoru): change to num_env if there is way to pass it in
    random_choice=False,
    activate_contact_sensors=True,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        kinematic_enabled=True,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    ),
)

def get_random_object_multi_spawner_cfg_based_on_num_envs(num_envs):
    return sim_utils.MultiAssetSpawnerCfg(
    assets_cfg=[get_random_object_spawner_cfg() for _ in range(num_envs)],
    random_choice=False,
    activate_contact_sensors=True,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    ),
    collision_props=sim_utils.CollisionPropertiesCfg(
        collision_enabled=True,
        contact_offset=0.002,
    ),
)


def get_TaskObjCfgDict(num_envs, env_config, sim_device):
    TaskObjCfgDict = {
        "table_grid": RigidObjectCfg(
            spawn=get_table_multi_spawner_cfg_based_on_num_envs(num_envs, env_config, sim_device),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
            ),
        ),
        "hold_object": RigidObjectCfg(
            spawn=get_random_object_multi_spawner_cfg_based_on_num_envs(num_envs),
            # spawn=sim_utils.UsdFileCfg(
            #     activate_contact_sensors=True,
            #     usd_path="gr00t/rl/data/objects/simple/bottle.usd",
            #     scale=(1.0, 1.0, 1.0),
            #     rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            #     visual_material=sim_utils.PreviewSurfaceCfg(),
            #     collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0, )
            # ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.875),  # Initial position for the root of the articulation
                #rot=(1.0, 0.0, 0.0, 0.0),  # Quaternion rotation (w, x, y, z)
                #lin_vel=(0.0, 0.0, 0.0),  # Linear velocity
                #ang_vel=(0.0, 0.0, 0.0),  # Angular velocity
            ),
        ),
        "front_object": RigidObjectCfg(
            spawn=get_random_object_multi_spawner_cfg_based_on_num_envs(num_envs),
            # spawn=sim_utils.UsdFileCfg(
            #     activate_contact_sensors=True,
            #     usd_path="gr00t/rl/data/objects/simple/bottle.usd",
            #     scale=(1.0, 1.0, 1.0),
            #     rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            #     visual_material=sim_utils.PreviewSurfaceCfg(),
            #     collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0, )
            # ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(2.5, 0.0, 0.875),  # Initial position for the root of the articulation
                #rot=(1.0, 0.0, 0.0, 0.0),  # Quaternion rotation (w, x, y, z)
                #lin_vel=(0.0, 0.0, 0.0),  # Linear velocity
                #ang_vel=(0.0, 0.0, 0.0),  # Angular velocity
            ),
        ),
        "back_object": RigidObjectCfg(
            spawn=get_random_object_multi_spawner_cfg_based_on_num_envs(num_envs),
            # spawn=sim_utils.UsdFileCfg(
            #     activate_contact_sensors=True,
            #     usd_path="gr00t/rl/data/objects/simple/bottle.usd",
            #     scale=(1.0, 1.0, 1.0),
            #     rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            #     visual_material=sim_utils.PreviewSurfaceCfg(),
            #     collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0, )
            # ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(-2.5, 0.0, 0.875),  # Initial position for the root of the articulation
                #rot=(1.0, 0.0, 0.0, 0.0),  # Quaternion rotation (w, x, y, z)
                #lin_vel=(0.0, 0.0, 0.0),  # Linear velocity
                #ang_vel=(0.0, 0.0, 0.0),  # Angular velocity
            ),
        ),
        "tray_object": RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/TrayTable",
            spawn=sim_utils.UsdFileCfg(
                usd_path="gr00t/rl/data/objects/simple/tray-28-28-10-thick-15mm.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.002,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[2.5, -0.3, 0.675],
                rot=[1.0, 0.0, 0.0, 0.0],
            ),
        )
    }
    return TaskObjCfgDict


def get_TaskObjCfgDict_and_USDIdx_based_on_num_envs(num_envs, env_config, sim_device):
    TaskObjCfgDict = get_TaskObjCfgDict(num_envs, env_config, sim_device)
    return TaskObjCfgDict, USD_ASSET_IDX_LIST

def get_object_class_names():
    """Return the list of object class names corresponding to the loaded USD assets."""
    return USD_ASSET_CLASS_NAMES.copy()

def get_object_class_name_for_indices(indices):
    """Return the object class names for the given asset indices."""
    if isinstance(indices, (list, tuple)):
        return [USD_ASSET_CLASS_NAMES[idx] for idx in indices]
    else:
        return USD_ASSET_CLASS_NAMES[indices]
