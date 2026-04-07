# Installation

```
conda create -n hv10 python=3.10
pip install isaacgym-1.0rc4-py3-none-any.whl
```



Install Isaac Gym

```
wget http://104.237.154.204/newest/isaacgym-1.0rc4-py3-none-any.whl
pip install isaacgym-1.0rc4-py3-none-any.whl
```

Install the dependencies

```
pip install -r groot/rl/requirements.txt 
```

## Unit-Test-0 : Train basic G1 locomotion

```
HYDRA_FULL_ERROR=1 python groot/rl/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+domain_rand=NO_domain_rand \
+rewards=loco/reward_g1_locomotion_2realable \
+robot=g1/g1_29dof_anneal_23dof \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
num_envs=4096 \
project_name=TestIsaacGymInstallation \
experiment_name=G123dof_loco \
headless=True \
rewards.reward_penalty_curriculum=True \
rewards.reward_initial_penalty_scale=0.2 \
rewards.reward_penalty_degree=0.00003 +opt=wandb
```

```
HYDRA_FULL_ERROR=1 python rl/train_agent.py \
+simulator=isaacsim \
+exp=locomotion \
+domain_rand=NO_domain_rand \
+rewards=loco/reward_g1_locomotion_2realable \
rewards.reward_scales.penalty_action_rate=-0.04 \
rewards.reward_scales.penalty_feet_height=-10.0 \
+robot=g1/g1_29dof_anneal_23dof \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_singlestep_wolinvel_a2c \
num_envs=4096 \
project_name=TestIsaacGymInstallation \
experiment_name=G123dof_loco_wolinvel_a2c_isaacsim_-0.04actionrate_-10feetheight \
headless=True \
rewards.reward_penalty_curriculum=True \
rewards.reward_initial_penalty_scale=0.2 \
rewards.reward_penalty_degree=0.00002 +opt=wandb
```