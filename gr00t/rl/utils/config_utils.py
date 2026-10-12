# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math

from loguru import logger
from omegaconf import OmegaConf, open_dict


def configure_headless_physics(config):
    """Keep train and eval on the same rendering-free simulator contract."""
    no_render_headless = (
        config.headless
        and not config.simulator.config.render_results
        and not config.simulator.config.cameras.enable_cameras
    )
    if config.simulator.config.get("physics_only", False) and not no_render_headless:
        raise ValueError(
            "physics_only requires headless=true, render_results=false, "
            "and cameras.enable_cameras=false"
        )
    with open_dict(config.simulator.config):
        config.simulator.config.disable_visual_markers = no_render_headless
        config.simulator.config.disable_visual_materials = no_render_headless
    return no_render_headless


def register_rl_resolvers():

    try:
        OmegaConf.register_new_resolver("eval", eval)
        OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
        OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
        OmegaConf.register_new_resolver("sqrt", lambda x: math.sqrt(float(x)))
        OmegaConf.register_new_resolver("sum", lambda x: sum(x))
        OmegaConf.register_new_resolver("ceil", lambda x: math.ceil(x))
        OmegaConf.register_new_resolver("int", lambda x: int(x))
        OmegaConf.register_new_resolver("len", lambda x: len(x))
        OmegaConf.register_new_resolver("sum_list", lambda lst: sum(lst))
    except Exception as e:
        logger.warning(f"Warning: Some resolvers already registered: {e}")
