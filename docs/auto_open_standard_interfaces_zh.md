# AutoOpen 标准编排接口

## 阶段名称

阶段采用动作名称；名称不代表已建立接触，也不代表另一种控制接口。数值ID固定，运行时不提供旧名称别名；历史采集文件保持原样，新日志与入口参数使用下列名称。

| ID | 名称 | 行为 |
|---|---|---|
| 1 | `APPROACH_HANDLE` | 底盘到握把操作站位 |
| 2 | `MOVE_TO_PREGRASP` | 避让把手后到预抓取点 |
| 3 | `OPEN_GRIPPER` | 张开夹爪 |
| 4 | `INSERT_AROUND_HANDLE` | 前伸使把手进入夹爪槽内 |
| 5 | `GRASP_HANDLE` | 闭爪并形成稳定夹持 |
| 6 | `PRESS_HANDLE` | 下压把手释放锁舌 |
| 7 | `OPEN_DOOR_WITH_HANDLE` | 握住把手绕门轴推/拉出门缝 |
| 8 | `RESET_HANDLE` | 把手旋转复位 |
| 9 | `RELEASE_HANDLE` | 张爪并解除上下爪对把手的约束 |
| 10 | `WITHDRAW_FROM_HANDLE` | 从把手处后撤 |
| 11 | `STOW_ARM_AND_CLOSE_GRIPPER` | 收臂到收纳位并闭爪 |
| 12 | `MOVE_TO_PANEL_STATION` | 推门链路到首次推板站位 |
| 13 | `MOVE_TO_PANEL_STANDOFF` | 夹爪到门板外预留间隙位置，不要求接触 |
| 14 | `PUSH_PANEL_OPEN` | 推门板直至门角足够或需要换站 |
| 15 | `STOW_ARM_AFTER_PUSH` | 推门链路推板后收臂 |
| 16 | `ADVANCE_TO_NEXT_PUSH_STATION` | 推门链路前进到下一推板站位 |
| 17 | `PASS_THROUGH_DOORWAY` | 对准门框并穿门 |
| 18/19 | `SUCCESS` / `FAILURE` | 成功/失败终态 |
| 20 | `MOVE_TO_GAP_STATION` | 后退20 cm后到门缝操作站位 |
| 21 | `INSERT_INTO_DOOR_GAP` | 闭爪对准并伸入门缝 |
| 22 | `PRY_DOOR_OPEN` | 保持夹爪世界朝向，绕门轴拨门 |
| 23 | `STOW_ARM_AFTER_PULL` | 拉门链路拨门或推板后收臂 |
| 24 | `FOLLOW_WALL_TO_PANEL` | 沿预计算肩部轨迹贴近门板，推进受门角限制 |

## 分层

推门、拉门及后续任务使用相同的目标规划、控制执行和验收工具。阶段只负责选择目标、定义任务反馈及任务完成条件；不实现另一套速度控制器、逆映射、坐标转换或稳定窗口。

| 层 | 标准工具 | 阶段提供的内容 |
|---|---|---|
| 直接目标 | `arm_pose_target`、`base_target` | 目标位姿及参考坐标系 |
| 候选/路径规划 | `PosePlanningMixin._inverse_pose_bank`、`first_feasible`、`best_feasible` | 批量位姿、有效长度、任务几何条件及排序规则 |
| 底盘轨迹 | `BasePoseTrajectory.install/sample` | 路点、速度上限、可选进度门控 |
| 机械臂轨迹 | `PaddedPathBank.install_tensor/current/sample_at`、`arm_trajectory` | 已验证的位姿序列、长度及推进方式 |
| 机械臂关节 | `arm_joint_target` | 关节目标 |
| 夹爪 | `gripper_target` | 开合目标 |
| 稳定 | `update_robot_stability`、`TensorStabilityWindow` | 通用配置及任务附加条件 |
| 验收/诊断 | `record_acceptance`、`consecutive_steps` | 有名称、数值、比较符和阈值的任务条件，以及稳定结果 |

目标点不必是固定终点。根据门角/门速计算当前参考位姿，仍属于目标点模式；根据任务反馈选择预计算轨迹的采样位置，仍属于轨迹模式。反馈只修改参考或进度，不增加新的执行模式。

## 控制执行

`execution.py` 是唯一命令写入层；`TensorCommandBuffers.flush` 是唯一环境写入出口。阶段不能直接修改 `buffers.base/ee_cart/ee_quat/arm_joint/gripper`，也不能直接给环境写 teleop 命令。

### 底盘

`base_target(state, rows, xy, yaw, buffers, ...)` 统一把世界系平面位姿参考转换成低层机身系 `vx/vy/wz`。可选参数包括参考速度、验收目标、到位窗口启用、任务停止条件和运动允许条件。

`_track_station` 和 `_base_hold_command` 是标准站位/保持配置的便利调用，均进入同一个 `base_target`，不是独立控制器。轨迹的每个样本也进入同一个接口。

- 站位配置保留向目标方向的最低非零速度，以及反向前插入零指令。
- 保持、再接触和穿门保留既有反向确认/锁定配置。
- 任务停止条件与通用到位条件分开：例如接触点进入再接触范围、尾部越过门框，属于任务条件。
- 限速、到位迟滞、反向处理和稳定窗口由标准执行/验收工具维护，不能在新阶段复制控制公式。

### 机械臂

只有三种执行模式：

1. `arm_pose_target`：发布EE位姿目标；默认位置为世界系，可显式指定 `frame="goal_reference_local"`，姿态仍为世界系。统一转换为低层需要的height-invariant命令。
2. `arm_trajectory`：从GPU缓存轨迹逐帧取点并通过同一EE发布接口执行；`sample_at`支持由任务反馈选择小数索引。
3. `arm_joint_target`：发布关节目标，用于保持初始/收回姿态等。

夹爪开合通过独立的 `gripper_target`，不作为第四种机械臂运动模式。失败终态停止底盘并保持最后已发布的机械臂目标，不切换到另一种隐含控制器。

目标限速及圆弧参考生成使用共享设备数学工具；没有新增按实际跟踪误差暂停轨迹、路段交界等待或逐帧重新逆映射。

## 目标与候选规划

直接目标和候选目标最终使用同一GPU逆映射。任务几何生成器可以不同，但求解与选择接口相同：

- 单点视为长度1的位姿序列。
- 多候选或整段轨迹以批量张量提交；有效长度在GPU展开，不逐环境或逐路径点调用CPU求解。
- `solve_pose_targets` 是GPU无重力稳态逆映射的唯一标准包装。
- `first_feasible`按已排序候选选首个可行项，`best_feasible`按代价选可行项；门板推弧的完整性、最小进展及碰撞净空仍是任务条件。
- 候选可行不自动等于阶段完成。最终仍以真实状态完成任务与稳定验收，不以规划解替代实际到位。

## 完成验收

任务条件与通用稳定分开计算，再组合为阶段完成条件。

- 任务条件：位置/姿态误差、门角、夹持几何、再接触可达、尾部净空等。
- 机身稳定：共享 `update_robot_stability`，可配置线/角速度、窗口时长、站高、直立性、位置/yaw误差及门速。禁止各阶段再实现一个稳定窗口。
- 机械臂稳定：共享 `TensorStabilityWindow`，使用实际位姿历史跨度；窗口长度和位置/姿态跨度阈值可以配置。
- 组合条件的连续帧计数使用 `consecutive_steps`，不能用瞬时满足替代规定时长。
- `record_acceptance`统一记录每项实际值、阈值、满足帧数、连续窗口和独立阻塞情况。诊断是被动观察，不改变控制决策。

当前阶段参数沿用已验证值，没有为接口重构放宽阈值、碰撞规则或预算。

## 全阶段约束

`stage_interfaces.py` 将底盘模式限定为目标/轨迹，机械臂模式限定为EE目标/EE轨迹/关节目标，目标来源限定为直接/候选。初始化检查全部Stage均有接口声明；新增Stage必须同时定义接口契约。

结构测试检查FSM没有直接命令写入，也不直接调用底盘速度控制公式；执行和规划接口不得读取GPU张量到CPU。单环境和多环境使用同一实现，没有标量兼容控制路径。

旧CPU稳态求解器、路径worker及标量控制/规划实现已删除，不保留备用执行链路。初始化只解析URDF静态关节限位；日志与视频的CPU处理不属于控制求解路径。

## 验证状态

已在隔离worktree开发，冻结控制源码后，通过唯一 `run_auto_open.sh` 的canonical模式分别完成推门、拉门各1000 seed无相机回归。两批均为1700–2699、四卡各250环境，无采集异常。

| 模式 | 严格通过 | 本次墙钟 | 比较基准墙钟 | 通过seed集合 | 1000条逐episode验收报告 |
|---|---:|---:|---:|---|---|
| 推门 | 813/1000，81.3% | 2707秒，45分07秒 | 2713秒 | 完全一致 | 全部一致 |
| 拉门 | 818/1000，81.8% | 2980秒，49分40秒 | 3016秒 | 完全一致 | 全部一致 |

结构及功能测试485项通过；没有用流程完成率替代严格通过率，没有通过放宽验收、延长阶段预算或更换物理参数适配新接口。测试夹具同步到新接口边界，并补齐既有夹具缺失的配置，不改变运行参数。

结果目录分别为 `datasets/auto_open/canonical/auto_open_v1_seeds1700-2699_canonical_no_camera_20261026T185129` 和 `datasets/auto_open/canonical/auto_open_v1_seeds1700-2699_canonical_no_camera_20261026T193656`。各目录的 `acceptance_report.json`、`standard_interfaces_comparison.json`、`workflow_source.json`记录本次验收和配对对比。

上述大批次验证覆盖标准接口重构快照，不覆盖随后进行的未使用代码删除及格式清理；后者单独通过代码测试、格式和静态检查验证，未重跑大批次仿真。
