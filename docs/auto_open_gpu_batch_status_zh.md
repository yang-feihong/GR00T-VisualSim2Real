# AutoOpen 当前实现与基线

本文只描述当前工作区中 AutoOpen 的有效逻辑和已验收基线，不记录历史调试过程、废弃方案或中间实验。

## 代码快照

- 分支：`auto-open`
- 基准提交：`5fd3b19 Refine AutoOpen randomized retract control`
- 主状态机：`gr00t/rl/scripts/doorman_auto_open.py`
- 领域模块：`gr00t/rl/auto_open/`
- 唯一实验入口：`gr00t/rl/scripts/run_auto_open.sh`
- 当前 AutoOpen 相关测试：`141 passed`

`doorman_auto_open.py` 明确维护顶层阶段图；几何、规划、控制、生命周期、运行时和 I/O 分别由 `geometry.py`、`planning.py`、`control.py`、`lifecycle.py`、`runtime.py` 和 `io.py` 实现。单环境和多环境运行时共用 GPU 常驻的 `TensorDoorFSM`；`DoorController` 只在初始化和稀疏规划边界提供静态任务配置，不参与逐帧 `command/advance`。

## 固定运行入口

所有 AutoOpen 运行必须通过：

```bash
bash gr00t/rl/scripts/run_auto_open.sh
```

入口只有数据采集语义，不再维护能力回归、视频回归或标量采集等平行流程。`DATASET_ROOT` 必填；相机只是同一流程的配置：

| `CAMERA_MODE` | 采集内容 |
|---|---|
| `none` | 不创建 RGB 传感器，只记录状态、动作、接触与验收数据 |
| `dual` | 记录训练使用的机身和腕部两路 RGB |
| `six` | 记录腕部、机身及四路诊断 RGB |

固定运行参数包括：

- 每个 episode 最多 `4000` 个 50 Hz 控制步；
- 精确路径和 CONTACT_PANEL 逆解使用 worker 后端；
- 门板间隙使用 GPU 后端；
- CENTER_HANDLE 守卫回退使用 NumPy batch 后端；
- 输出目录必须不存在，脚本拒绝覆盖；
- 每张物理 GPU 恰好启动一个 Isaac Sim 进程；分配到该 GPU 的所有 seed 在同一进程中作为 vectorized env 运行；
- 可用 `SEEDS` 指定任意 seed 集合，或用 `SEED_START`/`NUM_EPISODES` 生成连续集合；不存在每卡 250 个槽位的上限；
- 每个 episode seed 与 GPU、rank、env 槽位、batch size 和相机模式解耦；
- `dual`/`six` 为每个 env 创建独立的 render-only Camera，逐 seed 应用确定性挂载和标定随机化；相机不注册为 physics scene sensor，不参与 reset、控制或物理推进。

同一个 seed 使用确定性的无界环境位置映射，不预留固定数量的空环境槽位。场景参数、机器人初态、任务目标、相机标定和策略输入行不依赖 GPU、rank、env 槽位或 batch size。相机模式不得改变控制或物理轨迹。

逐帧控制链使用统一的 batch-invariant 数学接口。受约束 DLS 使用固定顺序正规方程和 Cholesky 求解；四元数四维归约使用与原单环境 CUDA 内核一致的配对顺序；其余原生归约经过 CUDA batch-shape 精确性测试后保留原数值语义。该实现没有阶段、seed 或接触状态特判。GPU PhysX 开启 enhanced determinism，并固定为一个 dynamics partition。当前完整接触 episode 已证明同一 seed 在不同 batch size、env 槽位和物理 GPU 下逐帧逐 bit 一致。

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
- 控制器决定继续移动时，任何非零平移指令的模长至少为 `0.15 m/s`，不再另设距离启用阈值；
- 控制器决定继续转动时，任何非零 yaw 指令的绝对值至少为 `0.2 rad/s`，不再另设角度启用阈值；
- 反向命令需要连续确认，并经过零速锁定，避免短时正负翻转；
- `APPROACH` 和 `ALIGN_PANEL` 共用同一个到位状态机和阶段配置；
- 首次进入阶段配置的内阈值后发布严格零速度，并开始累计静止窗口；
- 静止期间允许漂移至外阈值以内；超过任一外阈值立即取消窗口并恢复目标追踪；
- `BASE_ADVANCE` 的再接触/肩部安全和 `THROUGH` 的整机三维稳定/直立/门速要求是阶段附加条件，不改变统一到位语义。

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

PUSH_OPEN 以 `20 deg/s` 为门铰链目标角速度，动态门角提前量上限为 `2 deg`，EE 位置目标每帧最多移动 `3 mm`。门板接触点超过机械臂舒适范围后，先收回机械臂，再让底盘前进，之后重新计算 CONTACT_PANEL 目标。

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

场景、门、机器人初态和收回目标始终按 episode seed 确定性采样。机器人初始采样语义对应前端鱼眼相机的 x/y/yaw，再换算到底盘初态。

可选的 `SEMANTIC_TARGET_NOISE=1` 在轨迹规划前启用已审核的任务目标随机化：APPROACH 门法向/切向各 `+-3 cm`，推门高度 `+-5 cm`，THROUGH 横向 `+-8 cm`、yaw `+-0.06 rad`。接触敏感的抓取深度、闭合方向和抓取姿态不随机化；预计算轨迹只扰动终点后重新规划，不逐 waypoint 加独立噪声。

`dual`/`six` 模式按 seed、按相机独立采样安装和标定误差：位置半径 `2 cm`、旋转最大 `15 deg`、fx/fy 共同尺度 `+-8%`、principal point 半径 `5 px`、fisheye k1-k4 相对 `+-10%`。Isaac Sim 不支持的 skew 固定记录为实际值 `0`。相机随机化只改变观测，不得改变控制或物理轨迹。

## 数据采集

采集器是被动观察者，不参与 AutoOpen 决策。数据阶段唯一由逐帧 FSM 状态映射：`APPROACH` 为 `approach`，`PREGRASP` 至 `RESTORE_GRIPPER` 为 `handle_press`，`ALIGN_PANEL` 至 `THROUGH` 为 `traversal`；末尾连续 50 帧实际稳定状态为 `terminal_hold`。

控制频率 `50 Hz`，物理频率 `200 Hz`。每个 shard 包含 58 维 observation、25 维 action、完整状态审计、PhysX substep 接触、门资产路径/hash、PhysX/碰撞参数、该 seed 的实际相机标定、机器可读验收报告和中文摘要；视频字段由 `CAMERA_MODE` 决定。

状态与实采语义一致：B2 线/角速度使用 trunk 坐标系；EE 只使用机械链已有的 `0.100 m` 工具点，不重复偏移；数据集工具姿态去除 URDF gripper-link 固定网格旋转。Cartesian 控制记为 `inactive=0, reset=0`，保持当前关节角为 `inactive=1, reset=0`，运动回 middle preset 为 `inactive=0, reset=1`，到达后恢复 inactive。

终止段底盘命令必须连续 50 帧严格为零，且实际底盘、机械臂、夹爪和门速度均满足稳定阈值。接触按脚-地面、夹爪-把手、夹爪-门板、门内部和异常环境接触分类；异常接触或分类穿透超限直接隔离。数据验收不通过即该 episode 不成功，不维护第二套“控制成功率”。

## GPU 张量运行时

AutoOpen 只有 `TensorDoorFSM` 一套运行时。阶段、计时、稳定窗口、控制状态和锁存量为 GPU 张量，每帧按 stage mask 批量生成命令并验收；不逐 env 调用 `DoorController.command()/advance()`。稀疏的精确 IK 和路径规划只在阶段边界批量请求。单环境和多环境、无相机和带相机均使用同一 FSM 和同一采集器。

控制、低层策略和 PhysX 物理保持在 GPU。严格数据采集不是纯 GPU 流程：GPU PhysX contact view 在每个 200 Hz 物理子步保存接触张量，控制周期末在 CPU 聚合真实接触并写 Parquet/JSON。当前实现采用以下固定性能约束：

- 整个 USD 场景只遍历一次，统一建立所有环境的 sensor/filter 索引；
- 接触池容量按环境数配置为每环境 `256` 条，而不是按每个 sensor 重复分配；运行时记录峰值，容量触顶立即失败，不允许静默截断；
- 每个物理子步只克隆 GPU tensor，不执行逐接触 `.item()`/`.tolist()` 同步；每个 50 Hz 控制周期结束后统一搬运并聚合；
- 250 个环境的目标 FK 和 joint-mode 每帧只批量计算/读取一次，不逐环境重复执行 float64 FK 或搬运整批 mask；
- dataset manifest 只引用各 shard 的验收报告，不嵌套复制完整报告。

当前已验证：

- 一次固定入口调用可将 4 个 seed 分配到 2 张 GPU；每卡只有一个 Isaac Sim 进程、进程内 2 个 vector env，最终聚合得到 4 个不重不漏的 shard；
- 单进程 2 env 无相机采集能够独立写出两个严格 shard；
- seed 1717 在 batch=1/env0、batch=2/env0、batch=2/env1 及 GPU0/GPU2/GPU3 上均为完整 `2426` 帧；raw audit 的 `45/45` 字段逐 bit 相同；
- 当前单环境结果与修改前 partition1 单环境基线同为 `2426` 帧，raw audit 的 `45/45` 字段逐 bit 相同，证明确定性修复没有改变原单环境行为；
- `none`、`dual`、`six` 的 seed 1717 均为完整 `2426` 帧，raw audit 的 `45/45` 字段逐 bit 相同，证明相机开关不改变控制和物理行为；
- 2 env 双相机的四路图像均非黑屏，两个 env 的相机世界位姿分别对应各自场景，像素哈希不同；
- 当前单 env 双相机两路、六相机六路视频均为 `2426` 帧；首帧均非黑屏，模式内像素哈希分别 `2/2`、`6/6` 唯一。

完整证据目录：

- batch=1/env0：`logs_rl/invariance_native_math_full_b1_20261015`；
- batch=2/env0：`logs_rl/invariance_native_math_full_b2_20261015`；
- batch=2/env1：`logs_rl/invariance_slot1_full_20261015`；
- GPU3：`logs_rl/invariance_gpu3_full_20261015`；
- dual/six：`logs_rl/invariance_current_dual_20261015`、`logs_rl/invariance_current_six_20261015`。

## 基线状态

当前无图像严格数据基线由唯一入口、四张 GPU、seed `1700–2699` 生成：

```text
CAMERA_MODE=none
SEMANTIC_TARGET_NOISE=0
NUM_EPISODES=1000
MAX_STEPS=4000
```

结果目录：`logs_rl/auto_open_damping_half_1000_noimage_gpu_fixed_20261015`

| 指标 | 结果 |
|---|---:|
| 请求/完成 | 1000/1000 |
| 严格数据验收通过 | 696/1000（69.6%） |
| 严格数据验收失败 | 304/1000 |
| 墙钟耗时 | 2676 s（44.6 min） |
| 四 rank 控制完成数 | 216 / 217 / 208 / 200 |
| 总数据帧 | 2,451,850 |

数据验收失败即失败。下表按每个失败 episode 的首个阻塞原因唯一归类，合计 304：

| 首要失败类别 | 数量 |
|---|---:|
| 异常环境/阶段外接触 | 81 |
| `TURN_HANDLE` | 50 |
| 接触穿透超限 | 37 |
| `PUSH_OPEN` | 24 |
| 机械臂目标短时往返 | 23 |
| `GRASP_HANDLE` | 22 |
| 底盘命令短时往返 | 14 |
| `APPROACH` | 11 |
| `CONTACT_PANEL` | 10 |
| handle_press 期间底盘移动 | 10 |
| `PREGRASP` | 9 |
| `OPEN_GRIPPER` | 4 |
| `BASE_ADVANCE` | 3 |
| Stage2 入口 q5 不连续 | 2 |
| `OPEN_AND_CLEAR` | 2 |
| `PUSH_ARM_RETRACT` | 1 |
| 穿越前 q6 未回正 | 1 |

四个 rank 的最慢项范围为：严格接触/状态补全 `1069–1113 s`，阶段边界精确规划 `582–603 s`，GPU 策略与物理 `314–350 s`，逐帧采集准备 `256–266 s`，最终 shard 写盘 `154–156 s`。因此当前 44.6 分钟不能表述为“纯 GPU 耗时”：控制与物理是 GPU 张量化的，剩余主要成本来自完整严格接触审计、精确规划和数据落盘。

与同一代码语义下的未优化严格采集相比，控制结果和严格通过数保持不变，接触审计逐值等价；当前 manifest 为约 `200 KB`，不再把每个 shard 报告重复嵌套到数 GB。seed 1700 的完整 `raw_audit` 与 `raw_contacts` 已同优化前逐表逐值一致，1700–1703 的非接触状态/动作、帧数和控制结果逐值一致。
