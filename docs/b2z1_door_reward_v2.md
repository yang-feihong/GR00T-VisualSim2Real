# B2Z1 六阶段开门 Reward 设计

## 设计边界

训练阶段依次为：

```text
WALK_TO_DOOR -> GRASP -> OPEN_LATCH -> REPOSITION -> PUSH_OPEN -> THROUGH
```

每个阶段都分别描述机器狗、机械臂和外部环境的实际物理状态。Reward 不读取 actor 命令是否接近脚本目标，不复刻 AutoOpen 轨迹，也没有隐藏子阶段。阶段编号以 6 维 one-hot 同时提供给 actor 和 critic。

每阶段有连续进度 `p_s in [0, 1]`。每步使用有符号势能差：

```text
r_progress = c_s * (p_s(t) - p_s(t-1))
```

靠近目标得正奖励，远离目标扣回奖励，因此往返运动不能刷分。离散验收仅用于阶段切换和一次性完成奖励，不承担主要学习信号。

通用组合函数为：

```text
B(q_1, ..., q_n) = 0.5 * mean(q_i) + 0.5 * min(q_i)
```

每个分量改善都有梯度，同时最弱分量获得额外权重。

## 0. WALK_TO_DOOR

机器狗目标：到把手中心沿门框法向、机器人所在侧 `0.77 m` 的操作工位；平面误差 `<0.16 m`，航向误差 `<0.08 rad`，并稳定。

机械臂目标：六个关节回到 middle preset：

```text
[-1.57, 1.60, -0.63, -1.10, 0.0, 0.0]
```

最大关节误差 `<0.05 rad`，最大关节速度 `<0.08 rad/s`。

外部环境目标：无；把手和门板无接触属于安全约束。

连续势场：

```text
q_pos = clip(exp(-d_base / 1.0) / exp(-0.16 / 1.0), 0, 1)
q_yaw = clip(exp(-e_yaw / 0.7) / exp(-0.08 / 0.7), 0, 1)
q_arm = mean_j clip(exp(-|q_j-q_middle_j| / 0.5) / exp(-0.05 / 0.5), 0, 1)
p_0 = B(q_pos, q_yaw, q_arm)
```

上述物理条件连续保持 10 步后完成。

## 1. GRASP

机器狗目标：继续处于第一阶段操作范围内。允许移动和转向微调，不要求静止。

机械臂目标：从 middle preset 自主学习抓住把手。没有离把手 `6 cm` 的 pregrasp 点，没有预设开爪角度，也不规定先平移、再前伸、再闭合的顺序。

外部环境目标：无；门把手和门铰链不要求在本阶段运动。

连续势场分量：

- `q_pos`、`q_yaw`：保持底盘操作范围。
- `q_pocket`：实际夹持槽中心到随机尺寸把手中心的距离。
- `q_orientation`：实际夹爪姿态到可形成双侧夹持姿态的误差。
- `q_enclosure`：两爪内表面相对实际把手半径、轴向有效段和门板法向的连续包络质量。
- 真实接触出现后，用双爪近期接触与相向接触质量提高 `q_enclosure`，但最终验收仍只认真实接触。

```text
q_pocket = clip(exp(-d_pocket/0.15) / exp(-0.020/0.15), 0, 1)
q_orientation = clip(exp(-e_orientation/0.5) / exp(-0.05/0.5), 0, 1)
p_1 = B(q_pos, q_yaw, q_pocket, q_orientation, q_enclosure)
```

完成条件连续 5 步：底盘在操作范围内，并满足严格双侧夹持，包括两爪在胶囊体有效段相向接触、夹持槽误差 `<20 mm`、法向误差 `<12 mm`、轴向误差 `<0.05 rad`、相对速度与夹爪速度 `<0.08`、每侧力 `<=50 N`、无轴座误触和门板接触。

## 2. OPEN_LATCH

机器狗目标：仍在操作范围内，允许全身协同调整。

机械臂目标：持续保持严格双侧夹持。

外部环境目标：锁舌缩回量 `>=0.02 m`，并且门角达到由门宽、门厚和 `2 mm` 净间隙几何计算出的门缝角。

连续势场：底盘位置、航向、夹持槽保持、夹爪轴向和双侧包络形成保持质量 `q_hold`；锁舌和门缝均为连续比例。

```text
q_latch = clip(latch_position / 0.02, 0, 1)
q_gap = clip(hinge_angle / gap_angle, 0, 1)
p_2 = 0.10*q_hold + 0.45*q_latch + 0.45*q_gap
```

验收帧必须同时满足底盘范围、严格双侧夹持、锁舌充分缩回和门缝形成，不能依靠历史状态通过。

## 3. REPOSITION

机器狗目标：到达新的推门安全工位，位置、航向和速度稳定。

机械臂目标：不勾把手、不碰门板，闭合夹爪并到达当前门板外侧推门准备区域。

外部环境目标：已形成的门缝不能丢失。

连续势场包括底盘位置/航向、EE 位置/姿态、夹爪闭合、底盘/EE 速度、把手接触力指数衰减、门板接触力指数衰减和门缝保持比例：

```text
q_release_handle = exp(-F_handle / 10)
q_release_panel = exp(-F_panel / 10)
q_gap_hold = clip(hinge_angle / gap_angle, 0, 1)
p_3 = B(all reposition components)
```

完成条件连续 15 步。Reward 不限定松爪、撤离、转腕、闭爪和底盘移动的先后顺序。

## 4. PUSH_OPEN

机器狗目标：保持肩部与旋转门板的安全间隙，可向前移动改善机械臂工作距离。

机械臂目标：作为主要执行器安全推门；可短暂脱离并重新接触，不要求持续接触。

外部环境目标：门持续朝打开方向运动，目标速度 `20 deg/s`，最终门角 `>=89 deg`。

连续势场：

```text
q_base_safe = exp(-max(0, 0.10-clearance_shoulder)/0.10)
q_angle = clip(hinge_angle / 89deg, 0, 1)
q_speed = exp(-|opening_speed-20deg/s|/(20deg/s))
p_4 = 0.10*q_base_safe + 0.20*q_push_pose + 0.55*q_angle + 0.15*q_speed
```

完成时还要求本阶段真实发生过夹爪-门板推门接触。门速超过 `30 deg/s` 连续 3 步、肩部侵入或任务力超过 `600 N` 会失败。

## 5. THROUGH

机器狗目标：沿穿门方向位移 `>1.0 m`，并稳定。

机械臂目标：严格回到 middle preset，而不是仅把 EE 收到机身附近。

外部环境目标：门角保持 `>=65 deg`，保证可安全通过。

连续势场：

```text
q_distance = clip(signed_through_distance / 1.0, 0, 1)
q_arm = mean_j clip(exp(-|q_j-q_middle_j|/0.5) / exp(-0.05/0.5), 0, 1)
q_door = clip(hinge_angle / 65deg, 0, 1)
p_5 = (q_distance + q_arm + q_door) / 3
```

完成条件连续 10 步：穿门距离、middle preset 误差与速度、门角和底盘稳定同时满足。

## 独立阶段训练

阶段切换验收通过的同一物理步会保存完整状态：机器狗 root/关节状态、机械臂与夹爪状态、门 root/关节状态、累计命令和任务缓冲。每个 env 保存最近 16 个合法快照；由于随机门几何与 env 绑定，禁止跨 env 移植。

可从随机模型参数单独训练阶段 `k`：

```bash
TRAINING_STAGE=k \
STAGE_INITIAL_STATE_SOURCE=/path/to/last.pt \
bash gr00t/rl/scripts/train_b2z1_door_open.sh
```

`k>0` 时，每回合从该 env 的阶段 `k` 合法初态库随机采样；这些状态均来自阶段 `k-1` 的真实验收事件。加载后环境会按当前代码重新检查上一阶段的瞬时物理验收条件，不合格的旧状态库会明确报错。固定阶段达到本阶段验收后直接结束回合，不继续执行下一阶段。阶段 one-hot 仍作为模型输入。
