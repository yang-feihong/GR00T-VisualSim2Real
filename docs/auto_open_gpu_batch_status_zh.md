# AutoOpen 当前版本说明

本文只维护当前推门、拉门的运行方式、核心实现、最近有效整批结果和已知问题，不维护历史修改记录。

## 验收口径与结果

成功率指严格数据验收通过率，不是开门完成率。异常接触、穿透、控制指令异常或其他验收失败，即使流程到达 SUCCESS，也不计成功。以下均为 canonical 无相机测试，不包含补图耗时；图像不参加这些批次的验收，其他验收保持完整。

| 模式 | Seed 闭区间 | 严格通过 | 严格失败 | 入口墙钟耗时 |
|---|---|---:|---:|---:|
| 推门 | 1700–2699 | 813/1000，81.3% | 187 | 2707 s，45分07秒 |
| 拉门 | 1700–2699 | 818/1000，81.8% | 182 | 2980 s，49分40秒 |
| 拉门，独立验证区间 | 4700–5699 | 841/1000，84.1% | 159 | 2975 s，49分35秒 |

每批均为四卡、每卡单进程250个环境，1000个唯一 seed、1000份 metadata 和 shard 验收报告，无采集异常。

推门和拉门1700–2699均已在统一标准接口的同一源码快照下完整复验；各自通过seed集合和1000条逐episode验收报告与重构前完全一致，耗时分别减少6秒和36秒。结构及功能测试485项通过。独立4700–5699是当前拉门规则的补充验证结果，未在本次接口重构后重新运行，不冒充本次源码快照的独立在线复验。

## 严格失败分布

按每条数据的首个拒收原因互斥分类；每列合计等于该批严格失败数。同一条数据的后续问题不会在本表重复计数。

| 首要拒收原因 | 推门1700–2699 | 拉门1700–2699 | 拉门4700–5699 |
|---|---:|---:|---:|
| 异常环境接触 | 78 | 102 | 109 |
| ADVANCE_TO_NEXT_PUSH_STATION 底盘前进验收超时 | 42 | 0 | 0 |
| GRASP_HANDLE 握把验收超时 | 36 | 42 | 25 |
| STOW_ARM_AFTER_PULL 收臂验收超时 | 0 | 12 | 9 |
| MOVE_TO_PANEL_STANDOFF 规划失败 | 7 | 0 | 0 |
| OPEN_DOOR_WITH_HANDLE 初始推/拉开缝超时 | 7 | 9 | 7 |
| 压把手期间底盘移动 | 6 | 3 | 1 |
| MOVE_TO_PANEL_STATION 底盘对位超时 | 3 | 0 | 0 |
| 机械臂目标短时往返 | 3 | 11 | 1 |
| MOVE_TO_PREGRASP 路径规划失败 | 2 | 0 | 5 |
| 接触穿透超限 | 2 | 1 | 2 |
| 底盘指令短时往返 | 1 | 2 | 0 |
| **合计** | **187** | **182** | **159** |

推门78条异常接触的首次事件阶段：APPROACH_HANDLE 58、PASS_THROUGH_DOORWAY 14、FAILURE 5、ADVANCE_TO_NEXT_PUSH_STATION 1。MOVE_TO_PANEL_STANDOFF 阶段异常接触为0。FAILURE表示接触首次记录于失败切换阶段，不应误称为正常工作阶段的碰撞。

异常环境接触进一步按首次异常事件的 FSM 阶段拆分如下。仅统计上表以异常环境接触为首要拒收原因的样本，每条计一次，不是接触帧数或全批所有接触事件数。

| 首次异常接触阶段 | 推门1700–2699 | 拉门1700–2699 | 拉门4700–5699 |
|---|---:|---:|---:|
| APPROACH_HANDLE：接近门前站位 | 58 | 58 | 56 |
| MOVE_TO_PREGRASP：预抓取就位 | 0 | 1 | 2 |
| WITHDRAW_FROM_HANDLE：松开把手后回撤 | 0 | 30 | 39 |
| STOW_ARM_AFTER_PULL：拨门/推板后收臂 | 0 | 1 | 1 |
| MOVE_TO_PANEL_STANDOFF：前往推板目标 | 0 | 10 | 8 |
| PUSH_PANEL_OPEN：推门板 | 0 | 0 | 3 |
| ADVANCE_TO_NEXT_PUSH_STATION：推门循环底盘前进 | 1 | 0 | 0 |
| PASS_THROUGH_DOORWAY：穿门 | 14 | 2 | 0 |
| FAILURE：失败切换阶段 | 5 | 0 | 0 |
| **合计** | **78** | **102** | **109** |

## 运行阶段失败

本表只统计 FSM 运行失败，与上表严格数据首因不同，不能相加。某条运行失败的数据可能先因异常接触被拒收；流程完成的数据也可能严格拒收。

| 运行失败原因 | 推门1700–2699 | 拉门1700–2699 | 拉门4700–5699 |
|---|---:|---:|---:|
| ADVANCE_TO_NEXT_PUSH_STATION 超时 | 44 | 0 | 0 |
| GRASP_HANDLE 超时 | 36 | 42 | 25 |
| MOVE_TO_PANEL_STANDOFF 规划失败 | 7 | 0 | 0 |
| MOVE_TO_PANEL_STANDOFF 实际到位超时 | 0 | 6 | 5 |
| STOW_ARM_AFTER_PULL 超时 | 0 | 23 | 11 |
| OPEN_DOOR_WITH_HANDLE 初始开缝超时 | 7 | 9 | 7 |
| PUSH_PANEL_OPEN 夹爪接触力超过600 N | 5 | 0 | 0 |
| MOVE_TO_PANEL_STATION 超时 | 3 | 0 | 0 |
| MOVE_TO_PREGRASP 路径规划失败 | 2 | 0 | 5 |
| **合计** | **104** | **80** | **53** |

MOVE_TO_PREGRASP 路径在入口预计算，失败日志中的阶段名为 APPROACH_HANDLE；这里按实际失败任务归类。拉门两批均无 MOVE_TO_PANEL_STANDOFF 规划失败及推门板超速，但仍有实际到位超时，不能把规划成功等同执行成功。

## 统一入口

分支为 `auto-open`，唯一运行入口为 `gr00t/rl/scripts/run_auto_open.sh`。推门、拉门由 `DOOR_MOTION=push|pull` 选择，与采集模式相互独立。

| AUTO_OPEN_MODE | 行为 |
|---|---|
| official，默认 | 无相机完整采集 → 严格筛选 → 通过轨迹补双相机 → 严格复验 → 正式发布 |
| canonical | 无相机完整采集及严格验收，不进入后续图像链路 |
| debug | 六相机完整采集及严格验收，默认输出六视角拼接视频，不正式发布 |

四卡1000 seed无相机测试：

```bash
AUTO_OPEN_MODE=canonical DOOR_MOTION=push GPU_IDS=0,1,2,3 \
SEED_START=1700 SEED_END=2699 bash gr00t/rl/scripts/run_auto_open.sh
```

拉门将 `DOOR_MOTION` 改为 `pull`。四 seed 视频诊断使用同一入口，将模式改为 `debug`、区间改为1700–1703。

- 每张 GPU 一个 Isaac Sim 进程，进程内多环境；不固定物理批量为250。
- 推门默认最多4000控制步，拉门16000步，各阶段另有独立预算。
- 控制50 Hz，物理200 Hz。阶段进度在运行日志中输出。
- canonical、debug、正式数据分别位于 `datasets/auto_open/canonical/`、`debug/`、`releases/`。
- 正式名称为 `auto_open_v1_seeds<start>-<end>_episodes<accepted>_dual_camera_<YYYYMMDD>`；各次采集并列保存，不维护 latest 指针。
- 正式补图逐帧恢复 canonical 记录的机器人、关节和门状态，不重跑策略、FSM 或物理控制。

## 标准接口

所有阶段使用同一套基础工具编排，详见 [标准编排接口](auto_open_standard_interfaces_zh.md)。

- 目标来源：直接目标，或GPU批量候选/路径规划；候选选择与逆映射共用标准工具。
- 机身执行：目标位姿，或预计算位姿轨迹；均进入同一个 `base_target` 跟踪器。
- 机械臂执行：`arm_pose_target`、`arm_trajectory`、`arm_joint_target`；夹爪使用独立 `gripper_target`。
- 通用验收：共享机身稳定、机械臂位姿历史稳定、连续帧窗口及逐条件诊断，阶段只组合任务要求与稳定结果。
- `stage_interfaces.py`声明全部阶段的标准模式；结构测试禁止阶段直接写低层命令。`execution.py`是唯一命令写入层，`TensorCommandBuffers.flush`是唯一环境写入出口。
- 门速反馈或门角门控只决定目标/轨迹进度，不另设控制模式。控制、规划保持GPU张量，不新增CPU数值求解或逐环境控制路径。

## 控制流程

推、拉门共享前半段：

```text
APPROACH_HANDLE → MOVE_TO_PREGRASP → OPEN_GRIPPER → INSERT_AROUND_HANDLE → GRASP_HANDLE
→ PRESS_HANDLE → OPEN_DOOR_WITH_HANDLE → RESET_HANDLE
→ RELEASE_HANDLE → WITHDRAW_FROM_HANDLE → STOW_ARM_AND_CLOSE_GRIPPER
```

三个旋转依次绕把手轴、门轴、把手轴；第二段按任务方向前推或后拉。

推门后半段：

```text
MOVE_TO_PANEL_STATION → MOVE_TO_PANEL_STANDOFF → PUSH_PANEL_OPEN
门角不足且超出舒适区：STOW_ARM_AFTER_PUSH → ADVANCE_TO_NEXT_PUSH_STATION → MOVE_TO_PANEL_STANDOFF
门角足够：收臂 → PASS_THROUGH_DOORWAY → SUCCESS
```

拉门后半段：

```text
MOVE_TO_GAP_STATION → INSERT_INTO_DOOR_GAP → PRY_DOOR_OPEN
→ STOW_ARM_AFTER_PULL → FOLLOW_WALL_TO_PANEL → MOVE_TO_PANEL_STANDOFF → PUSH_PANEL_OPEN
门角不足：STOW_ARM_AFTER_PULL → FOLLOW_WALL_TO_PANEL → MOVE_TO_PANEL_STANDOFF
门角足够：收臂 → PASS_THROUGH_DOORWAY → SUCCESS
```

首次拨门要求至少30度；之后复用推板，不重复首次拨门。开门角阈值89度，但达到角度不等于数据成功。PASS_THROUGH_DOORWAY 在本阶段内先对准门框再穿门，不另设拉门对准门框阶段。

### 底盘与轨迹

- APPROACH_HANDLE 名义距离0.85 m；夹爪和把手高度区间接近时，按初始实际尖端前伸加2 cm增大距离，只增不减。
- 统一底盘位置内/外阈值2/10 cm，yaw内/外阈值0.05/0.15 rad，稳定窗口25帧；各阶段保留任务附加条件。
- 速度命令上限0.4 m/s、0.6 rad/s；非零平移至少0.15 m/s，非零转动至少0.2 rad/s。
- 底盘预计算轨迹默认线速度上限0.25 m/s、角速度上限0.40 rad/s。逐帧取点，结束后保持终点；不按实际跟踪误差暂停，不在路段交界额外等待。
- 首次横移的后退20 cm与前往站位属于同一预计算轨迹。当前站位局部采样区域为 x=0.535–0.555 m、y=0.630–0.650 m、yaw=-132至-126度，按门方向镜像映射；32个候选，最多四轮选站。插入目标要求在0.6 m内。
- FOLLOW_WALL_TO_PANEL 的靠墙前肩先沿墙移动，随后保持到门轴半径；门轴在左/右时半径分别0.40/0.60 m。朝向按 arcsin 几何规则计算，机身宽度额外加20 cm。轨迹预计算不依赖门角，发布点仅在其夹角小于实际门角时前进，索引不后退。
- PASS_THROUGH_DOORWAY 的终态要求尾部越过门框1 m，同时满足底盘、机械臂、夹爪和门的稳定条件。

### 机械臂与选点

- 低层 whole-body controller 接收 height-invariant EE 命令，不是上层直接下发关节角。不能把底盘高度变化简单等同世界系目标随之移动。
- 轨迹在阶段入口调用统一GPU无重力稳态逆映射，路径点批量计算；执行时发布缓存的EE轨迹，不逐帧重新 inverse。
- MOVE_TO_PREGRASP 使用预计算避把手轨迹；pitch候选及轨迹可行性在入口批量验证。
- MOVE_TO_PANEL_STANDOFF 共用连续区域生成规则：将门板接触线与当前0.70 m平面工作空间相交，保留板边、距门轴至少20 cm和把手避让要求；每个yaw取10个径向位置。yaw共7个，范围±30度。终点夹爪几何距门板3 cm。
- 拉门使用0.80–1.20 m、5 cm步长的9个高度，共630候选；保留至少5度有效推弧要求及现行选择次序。
- 推门使用0.80–1.30 m、5 cm步长的11个高度，共770候选；优先可用推弧，不以至少5度拒绝全部短弧；验证连续到位路径及站立高度裕量，必要时使用通用先旋转第一关节再展开的安全过渡备选。全部仍转换成EE轨迹并检查逆映射精度、关节裕量、整臂门板净空及速度。
- PUSH_PANEL_OPEN 的超前量定义在夹爪指定尖端，动态上限4度，门速反馈减小超前量；EE位置每周期最多变化3 mm。首次拉门拨门保持夹爪世界系方向固定，不随门板转向。

## 运行与数据边界

控制、策略、物理、候选和轨迹规划使用GPU张量。严格接触审计、数据打包、Parquet/JSON落盘以及视频处理包含CPU工作，因此不称整套采集为纯GPU耗时。

单环境、多环境及相机模式共用 TensorDoorFSM。场景随机化和控制数学不依赖rank或环境槽位；但共享GPU PhysX场景跨批量接触求解不能保证逐bit一致。正式补图恢复已记录状态，不依赖重新仿真的跨批量一致性。

## 当前主要问题

- 推门：异常接触78条、BASE_ADVANCE超时42条、握把超时36条是严格失败主体。CONTACT_PANEL仍有7条规划失败，表示当前候选和严格检查未找到可用路径，不代表完整六自由度空间无解。该阶段实际到位超时及异常接触均为0。
- 拉门：异常接触为主要损失，其次握把与收臂。CONTACT_PANEL候选规划在两批均通过，但实际到位仍有6/5条超时；候选终点合格不能证明整个执行过程合格。
- 以上是当前结果边界，不以流程完成、离线可行或测试通过替代严格数据验收，也不宣称所有阶段已无回归或无碰撞。

## 结果证据

以下路径均相对仓库根目录。

| 批次 | canonical结果目录 | 汇总证据 |
|---|---|---|
| 推门1700–2699 | `datasets/auto_open/canonical/auto_open_v1_seeds1700-2699_canonical_no_camera_20261026T185129` | `acceptance_report.json`、`standard_interfaces_comparison.json`、`workflow_source.json` |
| 拉门1700–2699 | `datasets/auto_open/canonical/auto_open_v1_seeds1700-2699_canonical_no_camera_20261026T193656` | `acceptance_report.json`、`standard_interfaces_comparison.json`、`workflow_source.json` |
| 拉门4700–5699 | `datasets/auto_open/canonical/auto_open_v1_seeds4700-5699_canonical_no_camera_20261025T224057` | `acceptance_report.json`、`panel_region_summary.json`、`workflow_source.json` |

推门、拉门同快照回归的源码指纹：`667f2a82db0a3c90a96873eb0c001bfc6881216f2c9e66a53c59f32edc6abf67`。独立拉门区间指纹：`9822a3e747230854bb1b438bef1b2e7fff56b953468ebc4361d8873e46482842`。具体采集快照以各目录的 `workflow_source.json` 为准。

验证边界：以上大批次覆盖标准接口重构快照；工作区另外清理了未使用的CPU求解器、旧控制路径并统一格式，这部分以代码测试及静态检查验证，尚未重新运行1000 seed。静态URDF关节上下限及margin与旧模型逐位一致；不能把采集快照指纹视为清理后工作区的指纹。
