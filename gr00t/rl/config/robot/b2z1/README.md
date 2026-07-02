# B2Z1 Door Debug Flow

本文档记录 B2Z1 接入 door opening 任务后的最小分阶段验证流程。原则是：训练任务、场景和 reward 继续复用原代码库；机器人资产使用离线转换后的 USD；高层只向低层提供 10 维接口。

## 10 维高层接口

高层 action 的顺序固定为：

```text
[vx, vy, omega_z, ee_dx, ee_dy, ee_dz, ee_droll, ee_dpitch, ee_dyaw, gripper]
```

其中：

```text
vx, vy, omega_z
```

是机身速度/角速度指令；

```text
ee_dx, ee_dy, ee_dz, ee_droll, ee_dpitch, ee_dyaw
```

是机械臂末端 6 自由度指令；

```text
gripper
```

是夹爪开合指令。

## 低层策略位置

默认低层策略路径是：

```text
models/b2z1_lowlevel/model_8000.pt
```

对应配置项在：

```text
gr00t/rl/config/robot/b2z1/b2z1.yaml
```

配置 key 是：

```text
robot.b2z1_command.lowlevel_policy_path
```

运行时也可以覆盖：

```bash
robot.b2z1_command.lowlevel_policy_path=/absolute/path/to/model_8000.pt
```

当前 loader 支持 `visual_wholebody` 训练保存的普通 checkpoint，也兼容 TorchScript/JIT deployment policy。

## 阶段 1：URDF 离线转换为 USD

仿真运行时继续走现有 USD 加载路径，不在 `IsaacSim._setup_scene()` 里新增 URDF runtime 分支。

使用 Isaac Sim Python 执行：

```bash
./isaac-sim/python.sh gr00t/rl/scripts/convert_b2z1_urdf_to_usd.py
```

成功标准：

```text
gr00t/rl/data/robots/b2z1/b2z1.usd
```

文件生成，并且脚本没有报 URDF mesh、IsaacLab importer 或 USD 写入错误。

## 阶段 2：只验证 B2Z1 能导入仿真器

这一步先不加载低层策略，只检查 B2Z1 USD、joint/body 命名、door 场景和 env wiring 是否能正常跑起来。

```bash
./isaac-sim/python.sh gr00t/rl/scripts/debug_b2z1_fixed_command.py \
  +exp=wbmanip/door_open_b2z1_lstm \
  num_envs=1 headless=False \
  robot.b2z1_command.lowlevel_policy_path=null \
  +fixed_command='[0,0,0,0,0,0,0,0,0,0]' \
  +num_debug_steps=200
```

成功标准：

- Isaac Sim 正常打开。
- 场景里能看到 B2Z1 和 door。
- 没有 missing USD、missing joint、missing body、frame transformer target 不存在等错误。
- 日志能持续打印 step、root、command 和 reward。

说明：命令里的 `+fixed_command` 和 `+num_debug_steps` 是 Hydra 新增配置项，不是 bash 的特殊语法。

## 阶段 3：加载低层策略

将低层策略放到默认路径：

```text
models/b2z1_lowlevel/model_8000.pt
```

然后先运行零速站立验证：

```bash
./isaac-sim/python.sh gr00t/rl/scripts/debug_b2z1_fixed_command.py \
  +exp=wbmanip/door_open_b2z1_lstm \
  num_envs=1 headless=False \
  +fixed_command='[0,0,0,0,0,0,0,0,0,0]' \
  +num_debug_steps=1000
```

成功标准：

- 日志中的 command 前三维速度为 0。
- B2Z1 在 viewer 里能稳定站立，不应在零速命令下持续抬腿或漂移。
- 没有低层 policy load、policy forward、action dimension 或 joint target dimension 报错。

## 阶段 4：逐项验证 10 维接口

### 原地转向

```bash
./isaac-sim/python.sh gr00t/rl/scripts/debug_b2z1_fixed_command.py \
  +exp=wbmanip/door_open_b2z1_lstm \
  num_envs=1 headless=False \
  +fixed_command='[0,0,0.3,0,0,0,0,0,0,0]' \
  +num_debug_steps=1000
```

成功标准：机器人产生稳定 yaw 转向。

### 夹爪开合

```bash
./isaac-sim/python.sh gr00t/rl/scripts/debug_b2z1_fixed_command.py \
  +exp=wbmanip/door_open_b2z1_lstm \
  num_envs=1 headless=False \
  +fixed_command='[0,0,0,0,0,0,0,0,0,1]' \
  +num_debug_steps=300
```

再测试反方向：

```bash
./isaac-sim/python.sh gr00t/rl/scripts/debug_b2z1_fixed_command.py \
  +exp=wbmanip/door_open_b2z1_lstm \
  num_envs=1 headless=False \
  +fixed_command='[0,0,0,0,0,0,0,0,0,-1]' \
  +num_debug_steps=300
```

成功标准：夹爪能按其中一个符号打开、另一个符号闭合。如果符号方向和预期相反，只需要调整 gripper command 到 `jointGripper` 的映射符号。

### 末端位置小位移

```bash
./isaac-sim/python.sh gr00t/rl/scripts/debug_b2z1_fixed_command.py \
  +exp=wbmanip/door_open_b2z1_lstm \
  num_envs=1 headless=False \
  +fixed_command='[0,0,0,0.05,0,0,0,0,0,0]' \
  +num_debug_steps=500
```

成功标准：机械臂末端对小的 `ee_dx` 指令有稳定响应。

## 常见失败定位

如果阶段 1 失败，优先检查 Isaac Sim/IsaacLab 环境、URDF mesh 路径和 converter API。

如果阶段 2 失败，优先检查：

- `gr00t/rl/data/robots/b2z1/b2z1.usd` 是否存在。
- `asset.usd_file` 是否仍然是 `b2z1/b2z1.usd`。
- `root_body_name`、`gripper_body_name`、`task_contact_body_names` 是否和 USD 里的 body 名一致。
- `dof_names` 是否和 USD articulation joint 名一致。

如果阶段 3 失败，优先检查：

- `models/b2z1_lowlevel/model_8000.pt` 是否存在。
- policy 是否为 TorchScript/JIT，或包含 `model_state_dict` 的训练 checkpoint。
- 低层 policy 期望 obs 维度是否等于 `robot.b2z1_command.lowlevel_policy_obs_dim`。
- policy 输出维度是否等于 `robot.b2z1_command.lowlevel_actions_dim`。

如果阶段 4 里底盘能动但机械臂不动，说明高层到低层的 10 维命令已经进入 bridge，但当前 bridge 的低层 observation/IK target 组织方式可能还没有和复制进来的低层策略部署接口完全一致。下一步应对齐 `gr00t/rl/third_party/visual_wholebody_lowlevel/legged_gym/envs/manip_loco/` 里的 deployment observation 和 target update 逻辑。
