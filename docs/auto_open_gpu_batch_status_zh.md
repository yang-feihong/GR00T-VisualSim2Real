# AutoOpen 当前实现与基线

本文只描述当前工作区中 AutoOpen 的有效逻辑和已验收基线，不记录历史调试过程、废弃方案或中间实验。

## 代码快照

- 分支：`auto-open`
- 基准提交：`5fd3b19 Refine AutoOpen randomized retract control`
- 主状态机：`gr00t/rl/scripts/doorman_auto_open.py`
- 领域模块：`gr00t/rl/auto_open/`
- 唯一实验入口：`gr00t/rl/scripts/run_auto_open.sh`
- 当前 GPU FSM 相关测试：`28 passed`

`doorman_auto_open.py` 明确维护顶层阶段图；几何、规划、控制、生命周期、运行时和 I/O 分别由 `geometry.py`、`planning.py`、`control.py`、`lifecycle.py`、`runtime.py` 和 `io.py` 实现。单环境和多环境运行时共用 GPU 常驻的 `TensorDoorFSM`；`DoorController` 只在初始化和稀疏规划边界提供静态任务配置，不参与逐帧 `command/advance`。

## 固定运行入口

所有能力回归、视频回归和数据采集必须通过：

```bash
bash gr00t/rl/scripts/run_auto_open.sh
```

脚本不接受临时拼接的命令行参数，只允许通过 `AUTO_OPEN_WORKFLOW` 选择已维护流程：

| workflow | 用途 | 必需输出变量 |
|---|---|---|
| `capability` | 无相机、多环境能力回归 | `OUTPUT_ROOT` |
| `semantic_collection` | 带已验收语义目标随机化的能力回归 | `OUTPUT_ROOT` |
| `video_regression` | 少量 seed 带相机视频回归 | `RUN_NAME` |
| `standard_collection` | 按 `DATA_STANARD.md` 采集双相机数据 | `DATASET_ROOT` |

固定运行参数包括：

- 每个 episode 最多 `4000` 个 50 Hz 控制步；
- 精确路径和 CONTACT_PANEL 逆解使用 worker 后端；
- 门板间隙使用 GPU 后端；
- CENTER_HANDLE 守卫回退使用 NumPy batch 后端；
- 输出目录必须不存在，脚本拒绝覆盖；
- 能力模式默认四卡，每卡 `250 env`；
- 每个 episode seed 与 GPU、rank、batch size 和相机开关解耦。

同一个 seed 使用确定性的无界环境位置映射和固定 256 行策略计算块。因此改变并行数量、GPU 分配或是否创建相机，不应改变该 seed 的初始场景和首帧低层动作。

## 顶层阶段图

所有阶段处于同一层级。每个阶段拥有独立步数预算，不存在跨阶段累计计数。

| 阶段 | 主要控制 | 主要验收 | 预算步数 |
|---|---|---|---:|
| `APPROACH` | 跟踪门相对底盘站位；机械臂保持初始关节姿态 | 底盘位置、yaw 和实测速度稳定 | 1000 |
| `PREGRASP` | 保持底盘；执行预抓取 EE 路径 | 路径完成，EE/底盘稳定且无异常接触 | 500 |
| `OPEN_GRIPPER` | 保持预抓取位姿；张开夹爪 | 预抓取仍有效且夹爪开度稳定 | 100 |
| `CENTER_HANDLE` | 夹爪张开，沿规划路径插入把手通道 | 把手在 pocket/channel 内对中且 EE 稳定 | 800 |
| `GRASP_HANDLE` | 保持 EE，持续下发夹爪闭合目标 `0 rad` | 双侧、带力且稳定的对向接触成立 | 400 |
| `TURN_HANDLE` | 保持夹持并沿压把手轨迹运动 | 锁舌释放、门开始运动且铰链速度足够低 | 900 |
| `RETURN_HANDLE` | 保持夹持并沿返回路径复位把手 | 把手回到水平附近并稳定 | 150 |
| `OPEN_AND_CLEAR` | 张开夹爪并离开把手 | 夹爪已张开且无把手接触 | 100 |
| `RETREAT_ARM` | 执行离开把手的回撤路径 | 路径完成且 EE 稳定 | 200 |
| `RESTORE_GRIPPER` | 执行随机化后的六自由度收回路径并闭合夹爪 | 实际 EE 到达同一几何目标、姿态正确且稳定 | 300 |
| `ALIGN_PANEL` | 保持机械臂关节角；底盘跟踪门板站位 | 底盘在门板站位稳定 | 400 |
| `CONTACT_PANEL` | 保持底盘；EE 渐进到门外 standoff | 实际 EE 达到门板相对目标并稳定 | 400 |
| `PUSH_OPEN` | 保持底盘；受速度和力约束地推门 | 门角足够，或机械臂超出舒适区需要换位 | 900 |
| `PUSH_ARM_RETRACT` | 保持底盘；收回机械臂 | 实际 EE 到达本次随机目标且姿态、速度稳定 | 300 |
| `BASE_ADVANCE` | 机械臂保持；底盘前往单调前进的安全再接触站位 | 接触点重新进入机械臂范围且底盘稳定 | 900 |
| `THROUGH` | 跟踪门框相对的尾部净空目标，同时恢复机械臂 | 尾部越过门框 1 m，底盘/机械臂/夹爪/门均稳定 | 1100 |
| `SUCCESS` | 保持最终命令 | 终态 | 无 |
| `FAILURE` | 保持失败命令 | 终态 | 无 |

正常主路径为：

```text
APPROACH
-> PREGRASP -> OPEN_GRIPPER -> CENTER_HANDLE -> GRASP_HANDLE
-> TURN_HANDLE -> RETURN_HANDLE -> OPEN_AND_CLEAR -> RETREAT_ARM
-> RESTORE_GRIPPER -> ALIGN_PANEL -> CONTACT_PANEL -> PUSH_OPEN
-> THROUGH -> SUCCESS
```

机械臂超出舒适区时，推门路径插入：

```text
PUSH_OPEN -> PUSH_ARM_RETRACT -> BASE_ADVANCE -> CONTACT_PANEL
```

该循环可以执行多次；每次进入 `PUSH_ARM_RETRACT` 都重新采样收回目标。

## 底盘控制

底盘移动阶段使用目标状态牵引，不直接长期发布固定速度。保持站位阶段在进入时锁存前一移动阶段的目标，而不是锁存带瞬时跟踪误差的当前位置。

全局约束：

- 平面速度上限：`0.4 m/s`；
- yaw 角速度上限：`0.6 rad/s`；
- 非零平移指令最低速度：`0.15 m/s`；
- 非零 yaw 指令最低角速度：`0.2 rad/s`；
- 反向命令需要连续确认，并经过零速锁定，避免短时正负翻转；
- 常规站位容差：位置 `0.12 m`，yaw `0.08 rad`；
- THROUGH 停车容差：位置 `0.08 m`，yaw `0.05 rad`。

`BASE_ADVANCE` 每步根据当前门板几何更新目标站位：目标尽可能前移，使门板接触点进入 `0.4 m` 再接触半径，同时满足前肩安全约束；再接触释放阈值为 `0.45 m`，目标带 `0.05 m` 提前量。目标纵向位置只能前进，不能后退；横向基准固定在门框中心。

`THROUGH` 的目标不是累计前进距离，而是机器狗尾部越过门框平面 `1.0 m` 的门框相对状态。终态同时要求实际速度稳定。

## 机械臂控制

### 控制模式

- `APPROACH`、`ALIGN_PANEL`、`BASE_ADVANCE` 等保持机械臂的阶段使用关节控制；
- EE 运动阶段使用 Cartesian 目标或预计算的稠密路径；
- EE 控制阶段保留位置和姿态限速，不再叠加会改变 Cartesian 轨迹的逐关节限速；
- 关节模式由低层统一限速；
- Cartesian/关节模式切换从当前实际关节状态连续进入，避免单帧目标跳变；
- 机械臂实际 actuator stiffness 与稳态模型共用同一配置。

### 预抓取与抓取

PREGRASP 在阶段入口批量尝试 `0/5/10/15/20 deg` 俯仰候选，保持固定爪相对把手的垂直预留量，选择首个精确可达解。路径只计算一次，执行期间不逐帧重算 IK。

CENTER_HANDLE 的语义是：

- 门板法向和把手轴向：把手中心对齐 `pocket_capture_center`；
- 竖直方向：把手中心对齐上下夹爪通道的 `channel_center`；
- 对中误差阈值：法向/轴向 `5 mm`，竖直 `10 mm`；
- 夹爪在把手轴向上必须保留端部余量：轴座侧至少 `20 mm`，自由端至少 `10 mm`。

GRASP_HANDLE 始终向上层记录闭合命令 `0 rad`。接触后是否停止继续加力属于夹爪低层闭环，反馈开度只记录为 observation，不能反写成监督命令。垂直平衡修正最大范围为 `30 mm`。

### 门板接触与推门

门板推门基准点使用门铰链侧底部为原点的固定偏移：沿门板切向 `0.4 m`、高度 `1.0 m`，不使用门板几何中心。

每次进入 CONTACT_PANEL 时：

1. 以基准点为中心生成上下左右 `0.2 m`、步长 `0.05 m` 的二维网格；
2. 一次批量执行精确 IK；
3. 选择离基准点最近的精确可达候选；
4. 夹爪指定尖端保持在门板外 `30 mm`；
5. 位置和姿态均从当前命令连续渐进，位置每周期最大变化 `3 mm`。

PUSH_OPEN 在门板接触阶段限制相对接近速度、接触力和门铰链角速度。门板接触点超过机械臂舒适范围后，先收回机械臂，再让底盘前进，之后重新计算 CONTACT_PANEL 目标。

### 收回目标随机化

RESTORE_GRIPPER 与 PUSH_ARM_RETRACT 共用中心目标，坐标为 B2Z1 goal-reference-local 的：

```text
[x, y, z, roll, pitch, yaw] = [0.30, 0.0, 0.30, -pi/2, 0.0, 0.0]
```

每次进入任一状态时独立采样：

- 位置：半径 `0.05 m` 球体内按体积均匀采样；
- 姿态：roll、pitch、yaw 分别在 `+-5 deg` 内均匀采样；
- 同一 episode 内不同入口不复用样本；
- 每批生成 `64` 个候选并一次执行 GPU batched IK；
- 精确可达标准：位置误差 `<=1e-4 m`、姿态误差 `<=2e-4 rad`、关节限位裕量 `>=2e-3 rad`；
- 选择采样顺序中的首个精确可达候选；整批无解则继续采样，最多检查 `4096` 个；
- 无解候选绝不下发，全部写入 episode metadata；
- 验收使用本次采样后的实际几何目标，不使用中心值或命令插值变量。

## 随机化

### 能力模式

`capability` 不启用额外语义目标噪声，但保留场景、门、机器人初态和收回目标的正式随机化。机器人初始采样语义对应前端鱼眼相机的 x/y/yaw，再换算到底盘初态。

### 语义采集模式

`semantic_collection` 在轨迹规划前对任务目标做 episode 级随机化：

| 变量 | 范围 |
|---|---:|
| APPROACH 门法向站位 | `+-3 cm` |
| APPROACH 门切向站位 | `+-3 cm` |
| 推门接触高度 | `+-5 cm` |
| THROUGH 横向目标 | `+-8 cm` |
| THROUGH yaw | `+-0.06 rad` |

接触敏感的抓取深度、夹爪闭合方向和抓取姿态默认不随机化。对预计算轨迹，随机扰动只作用于轨迹终点，随后重新规划整条平滑轨迹，不对每个 waypoint 单独加噪。

### 相机随机化

标准数据采集对每个相机独立采样，并记录实际生效标定：

- 安装位置：半径 `2 cm`；
- 安装旋转：最大 `15 deg`；
- fx/fy 共同尺度：`+-8%`；
- principal point：半径 `5 px`；
- fisheye k1-k4：相对 `+-10%`；
- skew 固定为 Isaac Sim 实际支持的 `0`；
- 标定文件记录渲染器实际采用的参数，而不是请求但未生效的参数。

相机随机化只改变观测，不改变控制场景、物理初态或 seed 对应关系。

## 数据采集

标准采集为被动观察者，不参与 AutoOpen 决策。物理阶段映射为：

| 数据阶段 | AutoOpen 阶段 |
|---|---|
| `approach` | `APPROACH` |
| `handle_press` | `PREGRASP` 至 `RESTORE_GRIPPER` |
| `traversal` | `ALIGN_PANEL` 至 `THROUGH` |
| `terminal_hold` | `SUCCESS` 后的稳定终止段 |

控制频率为 `50 Hz`，物理频率为 `200 Hz`，每个控制帧包含 4 个物理 substep。标准输出包括双相机视频、58 维 observation、25 维 action、完整状态审计、原始接触、门资产路径/hash、PhysX/碰撞参数、实际相机标定、机器可读验收报告和中文验收摘要。

机械臂监督模式：

- Cartesian 或正常任务控制：`inactive=0, reset=0`；
- 保持当前关节角：`inactive=1, reset=0`；
- 正在回到 middle preset：`inactive=0, reset=1`；
- 已到达 middle preset 并保持：`inactive=1, reset=0`。

终止段必须连续稳定，不能只依赖命令归零。当前阈值为：底盘线速度 `<0.05 m/s`、底盘角速度 `<0.05 rad/s`、机械臂关节速度 `<0.08 rad/s`、夹爪速度 `<0.05 rad/s`、门速度 `<2 deg/s`，持续 50 帧。

接触报告区分脚-地面、夹爪-把手、夹爪-门板、门内部接触和异常机器人-环境接触；力、冲量和穿透均来自 PhysX substep 数据。异常接触或超过分类阈值的穿透会进入隔离原因，不能自动判为合格。

当前标准采集烟测位于：

```text
logs_rl/tensor_fsm_standard_slew_smoke_20261013
```

seed 1700 控制成功并通过数据自动验收：`2774` 帧、双相机 `50 Hz`、状态/动作时钟抖动 `0 ns`、非零把手接触 `1820` 行、短时底盘/机械臂目标往返均为 `0`、隔离原因和例外均为空。单 seed 整体采集及验收耗时 `550 s`。

## GPU 张量运行时

AutoOpen 当前只有一套正式运行时：`TensorDoorFSM`。阶段、阶段步数、稳定窗口、控制状态和阶段间锁存量均为 GPU 常驻张量；每帧按 stage mask 批量生成底盘、Cartesian、关节和夹爪命令，并批量执行验收与阶段转换。

- 运行时不逐 env 调用 `DoorController.command()/advance()`；
- 常规能力回归不再每帧把完整控制状态复制到 CPU；
- stage mask 的几何行索引直接保留为 GPU 张量，不做 CPU list 往返；
- PREGRASP、CENTER_HANDLE、释放/回撤路径、随机收回目标和 CONTACT_PANEL 精确逆映射只在阶段边界形成稀疏批量请求；
- 单环境标准数据采集仍会将规范要求保存的 observation/action/contact 数据写回主机，这是数据输出，不参与控制决策；
- 单环境、多环境、视频和数据采集共用同一个 FSM，不存在标量控制分支或引擎选择开关。

视频 workflow 已用 seed 1700 验证成功，六相机独立视频和 mosaic 均正常写出到 `logs_rl/tensor_fsm_single_runtime_smoke_20261013_seed1700_video`。

测试覆盖底盘跟踪张量公式、独立路径索引、推门物理参考点、BASE_ADVANCE 重接触状态传递、PUSH_ARM_RETRACT 姿态渐变、固定入口和阶段契约。当前相关检查为 `28 passed`，Python 编译和 Shell 语法检查通过。

## 当前能力基线

固定口径：`capability`、无相机、四卡、每卡 `250 env`、seed `1700-2699`、每 episode 最多 `4000` 步。

结果目录：

```text
logs_rl/fsm_tensor_1000_slew_final_20261013
```

总体结果：

- 成功：`918/1000`；
- 成功率：`91.8%`；
- 总耗时：`1189 s`；
- metadata：`1000/1000`，seed `1700-2699`，无缺失或重复；
- 四个 rank：`231/250`、`229/250`、`230/250`、`228/250`。

失败阶段分布：

| 失败阶段 | 数量 |
|---|---:|
| `CONTACT_PANEL` | 18 |
| `PUSH_OPEN` | 18 |
| `APPROACH` | 12 |
| `GRASP_HANDLE` | 10 |
| `BASE_ADVANCE` | 4 |
| `PUSH_ARM_RETRACT` | 2 |
| `OPEN_GRIPPER` | 4 |
| `THROUGH` | 2 |
| `PREGRASP` | 3 |
| `TURN_HANDLE` | 3 |
| `ALIGN_PANEL` | 2 |
| `CENTER_HANDLE` | 2 |
| `RESTORE_GRIPPER` | 2 |

同口径迁移前验收值为 `916/1000、1865 s`。当前成功率提高 `0.2` 个百分点；同 seed 中共同成功 `872`，迁移前独有成功 `44`，当前独有成功 `46`，净增 `2`。总耗时减少 `676 s`，即 `36.2%`。

## 当前性能

四个 rank 的时间范围：

| 分项 | 当前 Tensor FSM | 迁移前标量运行时 |
|---|---:|---:|
| `command` | `72.5-76.9 s` | `762.4-777.0 s` |
| `advance` | `28.4-29.9 s` | `485.8-492.3 s` |
| 每帧完整状态回传 | `0 s` | `60.8-63.0 s` |
| policy + physics | `287.0-304.7 s` | `291.0-312.4 s` |
| 稀疏规划 | `656.9-695.3 s` | 约 `252-262 s` |

上层 FSM 的目标瓶颈已消除：`command/advance` 合计从每 rank 约 `1250 s` 降至约 `101-107 s`。当前最大耗时项转移为阶段边界的精确 IK/路径规划服务；它不在逐帧 FSM 内，但后续若继续提速，应针对该批量服务本身，不应修改控制阈值或物理语义。

## 输出与验收边界

能力回归每个 seed 写出 `metadata.json`，每个 rank 写出 `summary.json`，固定入口写出 `launcher_status.txt` 并检查 seed 缺失、重复及 rank 异常。任何后续控制语义修改都必须使用相同 seed、相同 4000 步预算和唯一固定入口重新验收。
