<div align="center">

# GR00T-VisualSim2Real


<table>
  <tr>
    <td align="center" valign="top" width="50%">
      <h3>Visual Sim-to-Real at Scale for<br>Humanoid Loco-Manipulation</h3>
      <a href="https://arxiv.org/abs/2511.15200"><img src="https://img.shields.io/badge/arXiv-2511.15200-b31b1b.svg" alt="VIRAL Paper"></a>
      <a href="https://viral-humanoid.github.io/"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="VIRAL Project Page"></a>
      <a href="https://gitlab-master.nvidia.com/ziwang/VIRAL/-/tree/main"><img src="https://img.shields.io/badge/Code-viral-lightgrey.svg" alt="VIRAL Code"></a>
      <br><br>
      <img src="./docs/viral-for-preview-v2-576P-ezgif.com-video-to-gif-converter.gif" width="100%">
    </td>
    <td align="center" valign="top" width="50%">
      <h3>Opening the Sim-to-Real Door for<br>Humanoid Pixel-to-Action Policy Transfer</h3>
      <a href="https://arxiv.org/abs/2512.01061"><img src="https://img.shields.io/badge/arXiv-2511.15200-b31b1b.svg" alt="DoorMan Paper"></a>
      <a href="https://doorman-humanoid.github.io/"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="DoorMan Project Page"></a>
      <a href="https://gitlab-master.nvidia.com/ziwang/VIRAL/-/tree/doorman"><img src="https://img.shields.io/badge/Code-doorman-lightgrey.svg" alt="DoorMan Code"></a>
      <br><br>
      <img src="./docs/doorman-teaser-576P-ezgif.com-video-to-gif-converter.gif" width="100%">
    </td>
  </tr>
</table>


</div>

<br/>

## Overview
This repository contains the application code for **VIRAL** (Visual Sim-to-Real for Humanoid Loco-Manipulation) and **DoorMan**. The system enables humanoid robots (e.g., Unitree G1) to perform complex tasks like opening heavy doors in the real world through a teacher-student simulation framework.

This repository contains the official code for:

- **VIRAL**: *Visual Sim-to-Real at Scale for Humanoid Loco-Manipulation*
  &nbsp; [Paper](https://arxiv.org/abs/2511.15200) &nbsp;|&nbsp; [Project](https://viral-humanoid.github.io/) &nbsp;|&nbsp; [Code](https://gitlab-master.nvidia.com/ziwang/VIRAL/-/tree/main)
- **DoorMan**: *Opening the Sim-to-Real Door for Humanoid Pixel-to-Action Policy Transfer*
  &nbsp; [Paper](https://arxiv.org/abs/2512.01061) &nbsp;|&nbsp; [Project](https://doorman-humanoid.github.io/) &nbsp;|&nbsp; [Code](https://gitlab-master.nvidia.com/ziwang/VIRAL/-/tree/doorman)

---


# VIRAL: Visual Sim-to-Real at Scale for Humanoid Loco-Manipulation

A reinforcement learning framework for humanoid robot loco-manipulation on the **Unitree G1** robot. The codebase supports:

- **Teacher Training** -- PPO-based policy training with privileged state observations
- **Student Training** -- Vision-based policy distillation (DAgger) from a trained teacher using RGB camera input
- **Evaluation** -- Evaluate trained teacher or student checkpoints, with optional ONNX export for deployment

Built on [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) (Isaac Sim 5.1), [TRL](https://github.com/huggingface/trl), and [Hydra](https://hydra.cc/) for configuration management.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Teacher Training (PPO)](#teacher-training-ppo)
  - [Teacher Evaluation](#teacher-evaluation)
  - [Student Training (DAgger)](#student-training-dagger-distillation)
  - [Student Evaluation](#student-evaluation)
- [Configuration](#configuration)
- [ONNX Export](#onnx-export)
- [Project Structure](#project-structure)
- [License](#license)
- [Citation](#citation)

## Prerequisites

- Ubuntu 22.04
- NVIDIA GPU with driver >= 535
- [Isaac Sim 5.1](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html)
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
- Conda or Mamba

## Installation

### 1. Create conda environment

```bash
conda create -n viral python=3.11 -y
conda activate viral
```

### 2. Install Isaac Sim 5.1

```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

pip install isaacsim==5.1.0.0 isaacsim-rl==5.1.0.0
```

### 3. Install Isaac Lab

Clone or download [Isaac Lab](https://github.com/isaac-sim/IsaacLab) and install:

```bash
pip install setuptools poetry-core flatdict

pip install --no-build-isolation -e <path-to-IsaacLab>/source/isaaclab
pip install --no-build-isolation -e <path-to-IsaacLab>/source/isaaclab_assets \
    -e <path-to-IsaacLab>/source/isaaclab_tasks \
    -e "<path-to-IsaacLab>/source/isaaclab_rl[all]"

pip install numpy==1.26.0
```

Verify the install:

```bash
python -c "import isaaclab; print(isaaclab.__file__)"
```

### 4. Install this package

```bash
cd <path-to-this-repo>
pip install -e .

pip install numpy==1.26.0   # pip may upgrade it; pin again
```

### 5. Verify installation

```bash
python -c "from groot.rl.envs.base_task.base_task import BaseTask; print('OK')"
```

## Usage

### Teacher Training (PPO)

Train a teacher policy using privileged state observations:

```bash
HYDRA_FULL_ERROR=1 accelerate launch --num_processes 1 \
    groot/rl/train_agent_trl.py \
    +exp=loco_manip/walk_stand_place_grasp_turn_homie \
    num_envs=48 \
    project_name=wsdpt_teacher
```

> **Tip:** Add `headless=False` to open the Isaac Sim GUI and watch training live.

<p align="center">
  <img src="./docs/viral-teacher-gif.gif" width="60%"><br/>
  <em>Teacher policy running in Isaac Sim</em>
</p>

| Argument | Description |
|---|---|
| `num_envs` | Number of parallel environments (higher = faster, more VRAM) |
| `project_name` | Weights & Biases project name |
| `headless` | `True` (default) for headless; `False` to open GUI |
| `env.config.reset_from_dataset.enable` | Reset from demonstration dataset |

### Teacher Evaluation

```bash
python groot/rl/eval_agent_trl.py \
    +checkpoint=logs_rl/<experiment_dir>/model_step_044500.pt
```

### Student Training (DAgger Distillation)

Train a vision-based student policy by distilling from a trained teacher:

1. Update the teacher checkpoint path in the experiment config:

```yaml
# config/exp/loco_manip/wsdpt_student_for_teacher_v8q8.002_resnet_rgb_delay.yaml
teacher_actor_path: logs_rl/<your_teacher_experiment>/model_step_XXXXXX.pt
```

2. Launch training:

```bash
HYDRA_FULL_ERROR=1 accelerate launch --num_processes 1 \@article{xue2025openingsimtorealdoorhumanoid,
  title={Opening the Sim-to-Real Door for Humanoid Pixel-to-Action Policy Transfer},
  author={Haoru Xue and Tairan He and Zi Wang and Qingwei Ben and Wenli Xiao and Zhengyi Luo and Xingye Da and Fernando Castañeda and Guanya Shi and Shankar Sastry and Linxi "Jim" Fan and Yuke Zhu},
  journal={https://arxiv.org/abs/2512.01061},
  year={2025},
</p>

### Student Evaluation

```bash
python groot/rl/eval_agent_trl.py \
    +checkpoint=logs_rl/<student_experiment_dir>/model_step_XXXXXX.pt
```

## Configuration

This project uses [Hydra](https://hydra.cc/) for configuration. Configs are composed from YAML files in `groot/rl/config/`. Override any value from the command line:

```bash
# Number of environments
... num_envs=16

# Reward weights
... rewards.reward_scales.tracking_lin_vel=1.0

# Training hyperparameters
... algo.config.actor_learning_rate=1e-4
```

### Experiment Tracking

Training logs to [Weights & Biases](https://wandb.ai/) by default:

```bash
wandb login
```

Checkpoints are saved to `logs_rl/<experiment_name>/` at intervals controlled by the `save_frequency` callback parameter.

## ONNX Export

During evaluation with `num_envs=1`, the policy is automatically exported as ONNX for deployment:

```bash
python groot/rl/eval_agent_trl.py \
    +checkpoint=<path_to_checkpoint.pt> \
    num_envs=1
```

The exported model is saved to `<experiment_dir>/exported/`.

## Project Structure

```
groot/rl/
├── train_agent_trl.py          # Training entry point (teacher & student)
├── eval_agent_trl.py           # Evaluation entry point
├── config/                     # Hydra YAML configs
│   ├── base.yaml               #   Base training config
│   ├── base_eval.yaml          #   Base evaluation config
│   ├── exp/loco_manip/         #   Experiment configs
│   ├── algo/                   #   Algorithm configs (PPO, DAgger)
│   ├── env/                    #   Environment configs
│   ├── robot/g1/               #   Robot configs (G1 43-DOF)
│   ├── rewards/                #   Reward function configs
│   ├── terrain/                #   Terrain configs
│   ├── obs/                    #   Observation configs
│   └── domain_rand/            #   Domain randomization configs
├── envs/                       # Environment implementations
│   ├── base_task/              #   Base task classes
│   └── loco_manip/             #   Loco-manipulation task
├── trl/                        # TRL-based trainers and modules
│   ├── trainer/                #   PPO and distillation trainers
│   ├── modules/                #   Actor-critic network modules
│   ├── callbacks/              #   Training callbacks
│   └── utils/                  #   Training utilities
├── agents/modules/             # Neural network building blocks
├── simulator/isaacsim/         # Isaac Sim interface
├── data/                       # Task data (robot assets, scenarios)
└── utils/                      # General utilities
```

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Copyright & Attribution

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This project includes third-party open-source software. Please refer to individual source files or `THIRD-PARTY-NOTICES.md` for specific licenses and copyright headers.

## Citation

If you find this work useful, please cite:

```bibtex
@article{he2025viral,
    title={VIRAL: Visual Sim-to-Real at Scale for Humanoid Loco-Manipulation},
    author={He, Tairan and Wang, Zi and Xue, Haoru and Ben, Qingwei and Luo, Zhengyi and Xiao, Wenli and Yuan, Ye and Da, Xingye and Castañeda, Fernando and Sastry, Shankar and Liu, Changliu and Shi, Guanya and Fan, Linxi and Zhu, Yuke},
    journal={arXiv preprint arXiv:2511.15200},
    year={2025}
}

@article{xue2025opening,
  title={Opening the Sim-to-Real Door for Humanoid Pixel-to-Action Policy Transfer},
  author={Xue, Haoru and He, Tairan and Wang, Zi and Ben, Qingwei and Xiao, Wenli and Luo, Zhengyi and Da, Xingye and Casta{\~n}eda, Fernando and Shi, Guanya and Sastry, Shankar and others},
  journal={arXiv preprint arXiv:2512.01061},
  year={2025}
}
}
```
