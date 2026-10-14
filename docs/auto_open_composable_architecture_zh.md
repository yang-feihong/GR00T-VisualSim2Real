# AutoOpen 可组合控制架构设计

## 1. 文档目的

本文定义 AutoOpen 后续重构的目标架构，不描述已经完成的实现，也不授权改变任何控制语义、目标值、阈值、阶段顺序或随机化分布。

本轮重构需要同时满足两个目标：

1. AutoOpen 主流程能够直接读出有哪些阶段，以及每个阶段的底盘、机械臂、夹爪命令、验收条件、预算和转移关系。
2. 控制能力被抽象为少量与阶段无关、与任务无关的标准接口。未来 AutoClose 或其他移动操作任务应通过重新组合这些接口实现，而不是复制 AutoOpen 的阶段分支。

本设计明确否定以下形式：

```python
base=TrackApproachStation()
arm=PushPanelWithSpeedLimit()
acceptance=StableBaseAtApproach()
```

这些名字仍然把一个阶段的完整私有逻辑封装进一个类，只改变了代码位置和写法，没有形成标准能力。

正确边界是：**有限的通用执行器 + 显式目标/约束数据 + 可组合验收条件 + 数据化阶段图**。

## 2. 当前实现的问题

当前主状态机位于 `gr00t/rl/scripts/doorman_auto_open.py`，约 3100 行，其中 `DoorController` 约 2700 行。一个阶段的完整语义分散在：

- `STAGE_SPECS`：只有文字说明、预算和后继阶段；
- `command()`：生成底盘、机械臂和夹爪命令；
- `advance()`：计算验收、更新阶段私有状态、规划下一段路径并执行转移；
- `DoorController.__init__()`：集中保存所有阶段的运行状态；
- `auto_open/control.py`、`planning.py`、`geometry.py`：已经抽出部分计算，但接口仍以修改 `DoorController` 字段为主。

因此当前存在以下结构性问题：

1. 阅读一个阶段必须同时跳转到多处代码。
2. 目标生成、命令执行和验收可能各自引用不同变量，容易出现“发布目标与验收参考点不一致”。
3. APPROACH、ALIGN_PANEL、BASE_ADVANCE 和 THROUGH 都是底盘目标位姿跟踪，但目前仍由阶段条件分支拼装。
4. Cartesian 跟踪、路径跟踪、关节保持、夹爪开闭、稳定窗口等能力没有统一生命周期接口。
5. 阶段入口规划和逐帧控制混在一起，难以确认一次性计算是否被错误地放进逐帧路径。
6. `DoorController` 同时承担任务图、运动控制、几何计算、规划、验收、日志和状态存储，职责过多。

已有的 `control.py`、`geometry.py`、`planning.py`、`lifecycle.py`、`runtime.py` 是可利用的第一层拆分，但还没有达到本文要求的模块边界。

## 3. 核心设计原则

### 3.1 阶段不能拥有私有控制实现

阶段只允许：

- 选择标准执行器；
- 为执行器提供目标和参数；
- 组合标准验收条件；
- 声明入口动作、预算和后继阶段。

不得新增 `ApproachController`、`ThroughController`、`CenterHandleController` 等一阶段一控制器的实现。

### 3.2 目标、执行和验收严格分离，但共享同一个目标记录

- **目标提供器**回答“名义任务目标是什么、随机化后物理上希望到哪里”。
- **执行器**回答“如何连续地把命令送到该目标”。
- **验收条件**根据实测状态和目标记录中明确指定的验收参考判断“是否到达”。

执行器可以使用逆映射后的命令目标，但验收不得拿逆映射命令代替物理意图目标。验收参考允许是随机化后的物理意图，也允许是名义任务目标，但必须在目标生成时明确选择并锁存，不能在验收代码中独立重算。

这是现有行为所必需的：APPROACH 当前向随机化站位发命令，但使用名义站位计算任务容差；RESTORE_GRIPPER 则必须对本次随机采样后的六自由度物理目标验收。二者都能由同一种目标类型表达，不需要阶段专用验收函数。

### 3.3 同一种运动必须复用同一个执行器

APPROACH 和 THROUGH 都必须使用同一个 `TrackBasePose`。二者只能在以下数据上不同：

- 目标位姿如何构造；
- 速度限制；
- 位姿容差；
- 稳定时间；
- 需要叠加的其他验收条件。

同理，PREGRASP、CENTER_HANDLE、CONTACT_PANEL 等 Cartesian 运动不能各自实现一套 EE 插值和限速。

### 3.4 目标是带坐标系和容差的类型，不是裸数组

所有目标必须明确：

- 物理意图值；
- 实际下发值；
- 坐标系；
- 有效环境 mask；
- 允许误差；
- 目标来源和随机化记录。

目标一经生成，执行和验收共享同一个目标对象，避免通过不同字段重新推导。

### 3.5 入口计算和逐帧计算明确区分

每个目标提供器和规划器必须声明更新策略：

- `ON_ENTRY`：进入阶段时计算一次并锁存；
- `EVERY_STEP`：依据实测状态逐帧更新；
- `ON_EVENT`：仅在显式事件发生时重新计算。

预抓取 IK、门板接触网格 IK、收回目标采样属于 `ON_ENTRY`。BASE_ADVANCE 的安全站位目标属于 `EVERY_STEP`，同时必须保持纵向目标单调前进。

### 3.6 单环境和多环境使用同一条 tensor 路径

所有接口接收 `env_mask`，输入和输出保留 batch 维度。单环境只是 batch size 为 1，不允许另建标量控制流程。核心逐帧路径不得引入逐 env Python 循环或 CPU/GPU 往返。

这里区分最终目标和当前现实：当前仿真、部分几何和低层策略是批量执行的，但上层仍为每个 env 创建一个 `DoorController`，每帧把完整控制状态拷到 CPU，再逐 env 调用 `command()` 和 `advance()`。重构不能把当前实现错误描述为纯 tensor FSM，也不能试图一步删除现有运行时。第 13 节给出可验证的迁移顺序。

## 4. 公共数据类型

### 4.1 只读状态快照

```python
@dataclass(frozen=True)
class TaskSnapshot:
    base: BaseState
    arm: ArmState
    gripper: GripperState
    articulations: ArticulationStateMap
    rigid_bodies: RigidBodyStateMap
    contacts: ContactState
    frames: FrameTree
    dt: float
```

快照集中定义控制和验收可读取的实测量。执行器、条件和规划器不得通过任意 `DoorController` 字段隐式取数。

### 4.2 类型化目标

```python
@dataclass(frozen=True)
class PoseGoal:
    nominal_pose: BatchedPose
    sampled_intent_pose: BatchedPose
    command_pose: BatchedPose
    acceptance_pose: BatchedPose
    frame: ReferenceFrame
    tolerance: PoseTolerance
    provenance: GoalProvenance

@dataclass(frozen=True)
class JointGoal:
    nominal_position: Tensor
    sampled_intent_position: Tensor
    command_position: Tensor
    acceptance_position: Tensor
    tolerance: Tensor
    provenance: GoalProvenance

@dataclass(frozen=True)
class ScalarGoal:
    nominal_value: Tensor
    sampled_intent_value: Tensor
    command_value: Tensor
    acceptance_value: Tensor
    tolerance: Tensor
    provenance: GoalProvenance

@dataclass(frozen=True)
class PathGoal:
    endpoint: PoseGoal
    command_waypoints: Tensor
    orientation_waypoints: Tensor
    timing: PathTiming
    provenance: GoalProvenance
```

- `nominal_*`：任务未加主动随机化时的目标；
- `sampled_intent_*`：应用一次性目标随机化后，物理上真正希望达到的目标；
- `command_*`：经过逆映射或低层接口转换后实际发布的控制值；
- `acceptance_*`：明确选择的验收参考，通常引用 `nominal_*` 或 `sampled_intent_*`，绝不引用 `command_*`。

四者及其来源必须保存在同一个对象中。这样既支持数据采集目标随机化，也不会再次出现命令参考点和验收参考点由两段代码分别计算的问题。

### 4.3 分通道命令

```python
@dataclass(frozen=True)
class TaskCommand:
    base: BaseCommand | None
    arm: ArmCommand | None
    gripper: GripperCommand | None

@dataclass(frozen=True)
class ArmCommand:
    mode: Literal["joint", "cartesian"]
    joint_target: Tensor | None
    cartesian_target: BatchedPose | None
    supervision: Literal["active", "inactive", "reset"]
```

每个阶段必须显式填写三个通道，不依赖上一帧命令“碰巧保留”。保持上一目标也必须通过明确的 `Hold...` 原语表达。

同一个 `TaskCommand` 是命令语义的唯一源，用于：

- 写入低层 teleop/控制 buffer；
- 做命令连续性审计。

物理步之后，运行时从低层返回值构造 `DispatchedCommandSnapshot`，记录经过低层裁剪和转换后真正执行的 base command、arm target 与 gripper target。数据集 action 由 `DispatchedCommandSnapshot` 的数值和 `TaskCommand.arm.supervision` 的语义共同生成；不能另行猜测一套监督命令。

机械臂控制模式和监督模式都必须由行为显式产生：保持当前关节角为 `inactive`；运动回 middle preset 期间为 `reset`；到达后保持 middle preset 为 `inactive`；Cartesian 任务控制为 `active`。采集器不再通过“关节目标是否刚好等于 middle preset”反推意图。

### 4.4 独立运行状态

状态按能力归属保存，而不是全部堆在 `DoorController`：

- `StageRuntimeState`：阶段、阶段步数、入口标志、转移原因；
- `BaseTrackingState`：锁存目标、停止锁、反向确认、稳定计数；
- `CartesianTrackingState`：命令位姿、路径索引、重规划事件；
- `GripperState`：命令目标和反馈；
- `ContactMotionState`：接触速度、力、进度和保护状态；
- `ConditionState`：各 `StableFor` 条件的连续计数；
- `TaskMemory`：跨阶段明确传递的语义结果，如选中的抓取目标和门板接触点。

## 5. 需要抽象的标准功能

### 5.1 目标构造器

目标构造器只计算物理目标，不执行控制。

| 接口 | 功能 | 当前用途 |
|---|---|---|
| `PoseInFrame` | 在指定坐标系中用固定六自由度偏移构造目标 | APPROACH、收回中心目标 |
| `LatchMeasuredPose` | 在入口锁存实测位姿 | 必须保持当前状态的通道 |
| `LatchPreviousGoal` | 继承并锁存上一个运动阶段的目标 | 抓取和推门期间保持底盘站位 |
| `PoseFromObjectGeometry` | 根据刚体几何、语义轴和固定偏移构造位姿 | 预抓取、把手对中、门板接触 |
| `PoseSatisfyingConstraints` | 从一组几何约束计算目标位姿 | 尾部净空、肩部安全站位 |
| `PoseOnArticulationOrbit` | 根据关节轴、轴心、接触点和目标关节位置计算刚体轨道上的 EE 位姿 | TURN_HANDLE、RETURN_HANDLE、PUSH_OPEN |
| `ApplyTargetRandomization` | 在规划前对一个目标做一次采样并记录 | 数据采集随机化、收回目标随机化 |
| `MonotonicGoalFilter` | 限制动态目标只能沿指定方向单调变化 | BASE_ADVANCE |
| `ComposePoseOffset` | 在明确坐标系中组合平移或旋转偏移 | standoff、插入深度、回撤距离 |

门、门把手、门框的尺寸和语义轴属于 `DoorGeometryModel`。它可以为上述通用构造器提供 frame、axis、surface 和 bounds，但不能直接发送控制命令。

候选集合也使用标准构造器：

| 接口 | 功能 | 当前用途 |
|---|---|---|
| `AngleSweepCandidates` | 围绕指定轴生成角度候选，并可固定一个世界参考点 | PREGRASP 的 `0/5/10/15/20 deg` 与高度耦合 |
| `PlanarGridCandidates` | 在给定平面的两个轴上生成二维网格 | CONTACT_PANEL 的 `+-0.2 m`、`0.05 m` 网格 |
| `RandomBallPoseCandidates` | 在球体位置和角度边界内按指定 RNG 批量采样 | 收回目标每批 64 个候选 |
| `LineCandidates` | 沿固定直线生成候选 | BASE_ADVANCE 门框中线安全搜索 |

候选构造器只返回 `PoseCandidateSet`；是否可达、误差阈值和最优候选选择统一交给 `SelectReachableCandidate`。

`PoseSatisfyingConstraints` 使用的约束同样是通用几何谓词：

| 约束 | 含义 |
|---|---|
| `BodyPointPlaneDistance` | 指定机身点到指定平面的有符号距离 |
| `LateralCoordinate` | 目标在给定轴上的投影值 |
| `YawEquals` | 目标 yaw |
| `PointReachableFromFrame` | 某世界点变换到目标机身/机械臂参考系后位于给定可达范围 |
| `BodiesClearOfSegment` | 一组机身碰撞代理与给定线段/平板保持最小距离 |
| `MonotonicAlongAxis` | 新目标沿指定轴不得小于已锁存目标 |

THROUGH 由前三项直接解出底盘目标；BASE_ADVANCE 在 `LineCandidates` 上依次应用后四项并选择最远安全前进点。约束对象不知道 APPROACH、THROUGH 或 BASE_ADVANCE 的阶段名称。

### 5.2 规划器

规划器把目标转换为完整路径或控制所需的命令目标，不负责阶段转移。

| 接口 | 功能 |
|---|---|
| `SolveCartesianPose` | 对一批六自由度目标执行 IK/稳态逆映射 |
| `PlanCartesianPath` | 从当前实测位姿到目标位姿生成限速稠密路径 |
| `PlanAxisRotationPath` | 围绕给定世界轴和轴心生成刚体旋转路径 |
| `PlanJointPath` | 从当前关节状态到目标关节状态生成连续路径 |
| `SelectReachableCandidate` | 对候选目标进行一次批量求解并按代价选取 |

路径规划必须明确使用的起点、终点、速度限制和重新规划策略。不能在每个 waypoint 上独立添加随机噪声。

### 5.3 底盘执行器

底盘只需要以下标准执行器：

| 执行器 | 输入 | 语义 |
|---|---|---|
| `TrackBasePose` | `PoseGoal`、速度/角速度上下限、停止与反向策略 | 根据目标位姿闭环生成平面速度和 yaw 角速度 |
| `HoldBasePose` | 锁存的 `PoseGoal`、同一组跟踪限制 | 使用 `TrackBasePose` 的同一内核保持站位 |
| `StopBase` | 无目标或停止策略 | 明确发布零速，仅用于确实不要求位置保持的终态/保护状态 |

`HoldBasePose` 不得拥有另一套控制算法；它只是以锁存目标调用 `TrackBasePose`。

### 5.4 机械臂执行器

| 执行器 | 输入 | 语义 |
|---|---|---|
| `TrackCartesianPose` | `PoseGoal`、位置/姿态速率限制 | 闭环跟踪单个 EE 位姿 |
| `FollowCartesianPath` | `PathGoal` | 按路径时序发布 EE 位姿 |
| `HoldCartesianPose` | 锁存或既有 `PoseGoal` | 使用 `TrackCartesianPose` 内核保持 EE 位姿 |
| `TrackJointPose` | `JointGoal`、关节速率限制 | 闭环跟踪目标关节角 |
| `FollowJointPath` | 关节路径 | 连续执行预计算关节轨迹 |
| `HoldJointPose` | 锁存或既有 `JointGoal` | 使用 `TrackJointPose` 内核保持关节角 |

接触动作仍使用 `TrackCartesianPose`，不再另造一个包含整套推门逻辑的机械臂执行器。PUSH_OPEN 的区别来自逐帧更新的 `PoseOnArticulationOrbit` 目标，以及显式组合的进度控制器、接触退让修正器和保护条件。这样执行层始终只负责位姿跟踪，推门、关门、按压等任务差异留在目标和约束层。

### 5.5 夹爪执行器

| 执行器 | 输入 | 语义 |
|---|---|---|
| `TrackGripperPosition` | `ScalarGoal`、速率限制 | 持续发布明确开度目标 |
| `FollowGripperPath` | 标量路径 | 与机械臂路径同步改变开度 |
| `HoldGripperPosition` | 锁存或既有目标 | 保持明确开度语义 |

“握住把手”在上层始终是闭合目标 `0 rad`。接触后限制继续施力属于夹爪低层控制，不得把反馈开度反写成上层命令。

### 5.6 目标修正器

少数动作需要基于反馈连续修正目标。修正器必须是显式、可组合和有界的，不能藏在阶段分支中：

| 修正器 | 功能 | 当前用途 |
|---|---|---|
| `BoundedAxisCorrection` | 根据标量误差沿指定轴缓慢修正，带总量和每步限幅 | GRASP_HANDLE 垂直平衡 |
| `ContactRetreatCorrection` | 接触力或穿透过大时沿接触法向退让 | PUSH_OPEN 保护 |
| `ProgressLeadController` | 根据被操作关节的速度/进度调整命令提前量 | PUSH_OPEN 门角跟踪 |

修正器只修改目标，不负责验收和阶段跳转。

`ProgressLeadController` 的输入必须是通用的关节位置、关节速度、目标速度和 PI/限幅参数；输出只是下一目标关节位置。随后由 `PoseOnArticulationOrbit` 转成 EE 位姿。它不能读取 `Stage.PUSH_OPEN` 或直接操作门对象。

### 5.7 领域测量器

验收条件本身保持通用；任务特有的几何量由只读测量器提供：

| 测量器 | 输出 |
|---|---|
| `BasePoseMetrics` | 目标位置误差、yaw 误差、线速度和角速度 |
| `PoseTrackingMetrics` | EE 位置误差、姿态误差和速度 |
| `HandleCaptureMetrics` | pocket 法向/轴向误差、channel 高度误差、抓取深度、两端余量、门板间隙 |
| `OpposedContactMetrics` | 两侧接触时效、接触力、是否对向、接触点语义位置 |
| `PanelContactMetrics` | 指尖到门板间隙、切向误差、插入轴误差、相对接近速度和接触力 |
| `TraversalMetrics` | 尾部净空、机身高度、upright、各系统速度 |

其中 `HandleCaptureMetrics`、`PanelContactMetrics` 和 `TraversalMetrics` 可以位于 AutoOpen 领域模块，因为它们依赖门资产语义；它们只产生 batched tensor，不拥有控制状态，也不执行转移。

### 5.8 验收条件

所有条件返回 batched boolean、数值诊断和失败原因，不直接修改阶段。

**基础条件：**

- `PoseWithinTolerance`
- `PositionWithinTolerance`
- `OrientationWithinTolerance`
- `JointPoseWithinTolerance`
- `ScalarWithinTolerance`
- `LinearSpeedBelow`
- `AngularSpeedBelow`
- `PathComplete`
- `ContactPresent`
- `ContactAbsent`
- `OpposedContactsPresent`
- `ContactForceWithinRange`
- `PenetrationBelow`
- `ObjectJointAbove/Below`
- `AxialMarginsAtLeast`
- `RobotUpright`
- `NoAbnormalContact`

**组合条件：**

- `AllOf`
- `AnyOf`
- `Not`
- `StableFor`
- `AfterMinimumDuration`

例如“位姿达到并连续稳定 15 帧”必须写为：

```python
StableFor(
    AllOf(
        PoseWithinTolerance(BASE, goal=stage_goal("base")),
        LinearSpeedBelow(BASE, 0.08),
        AngularSpeedBelow(BASE, limit),
    ),
    frames=15,
)
```

不得新增 `StableBaseAtApproach` 之类只服务一个阶段的条件。

### 5.9 机器人低层适配器

通用行为统一输出物理语义命令：世界系 Cartesian 位姿、关节目标、夹爪目标和底盘速度。B2Z1 当前低层接口所需的特殊编码由一个共享适配器完成：

```python
class B2Z1CommandAdapter:
    def encode(
        self,
        command: TaskCommand,
        snapshot: TaskSnapshot,
        env_mask: Tensor,
    ) -> LowLevelCommandBuffers: ...
```

该适配器负责：

- world EE position 到 goal-reference-local position 的变换；
- world quaternion 到低层 RPY delta 的转换；
- 当前 `arm_induced_pitch` 约定；
- joint/Cartesian control mode mask；
- 低层 command buffer 的 batched 写入；
- 全局低层裁剪前后的审计字段。

现有 `_ee_local_target`、`_ee_world_target`、`_grasp_orientation_delta`、arm mode 分支和 `_BatchedRunnerView.set_teleop_targets` 的有效语义应迁入此边界。适配器不能读取 AutoOpen 阶段名；AutoOpen、AutoClose 和其他任务共用同一个 B2Z1 命令语义。

## 6. 标准模块生命周期与真实时序

### 6.1 为什么需要两个转移时机

当前行为并非所有转移都发生在 physics step 之后：

- 常规到达、接触和稳定验收使用 step 后的实测状态；
- PUSH_OPEN 检测机械臂超出舒适区后，会在下一条命令生成前立即切到 PUSH_ARM_RETRACT；
- PUSH_ARM_RETRACT 达到目标、BASE_ADVANCE 可重新接触后，也是在命令生成过程中切换控制方式；
- CONTACT_PANEL 在门已继续转到 89 度时可直接进入 THROUGH。

若统一改成“验收后下一帧才切换”，会多执行一帧旧命令；若全部提前到命令前，又会使用错误的观测时刻。因此阶段转移必须明确标注相位：

```python
class TransitionPhase(Enum):
    PRE_COMMAND = "pre_command"
    POST_PHYSICS = "post_physics"
```

这不是两套 FSM。二者由同一个批量执行器按固定顺序处理，只是条件读取的快照时刻不同。

### 6.2 每个控制周期的固定顺序

```text
1. 取得 snapshot_t
2. 计算 PRE_COMMAND 测量量与转移条件
3. 对命中的 env mask 执行转移并初始化目标请求
4. 汇总本周期所需的稀疏规划请求，按 request kind 批量求解
5. 更新 EVERY_STEP 目标
6. 三个标准行为生成唯一 TaskCommand_t
7. TaskCommand_t 写入低层；采集器锁存 observation_t 和命令语义
8. 执行 policy + physics，取得 snapshot_t+1 和 DispatchedCommandSnapshot_t
9. 采集器用实际 dispatched 数值完成 action_t，运行时审计其与 TaskCommand_t 的关系
10. 批量计算 POST_PHYSICS 测量量与验收/保护条件
11. 对命中的 env mask 执行转移并汇总下一阶段入口规划请求
12. 批量完成这些规划，使下一周期可直接发出新阶段命令
13. 更新阶段时钟、诊断和超时
```

执行器必须规定保护条件优先级：全局安全失败 > 阶段失败 > 正常完成 > 超时。一个 env 同一相位最多提交一次转移。

### 6.3 模块接口

```python
goal_request = goal_provider.request(snapshot, memory, env_mask)
goal = goal_service.resolve(goal_request_batch)
goal = goal_provider.update(snapshot, goal, memory, env_mask)

command, behavior_state = behavior.step(
    snapshot, goal, behavior_state, env_mask
)
result, condition_state = condition.evaluate(
    snapshot, goal, condition_state, env_mask
)
```

- `request()` 声明计算，不在逐 env 调用中直接阻塞求解；
- coordinator 将同类 IK、路径和候选搜索请求汇总后一次批量提交；
- `update()` 只用于声明为 `EVERY_STEP` 的目标；
- `step()` 不得改变任务阶段；
- `evaluate()` 不得生成控制命令；
- transition executor 根据 condition result 和阶段图统一转移；
- 所有接口均处理任意数量环境的 tensor 和 mask。

当前通过 `DeferredSteadyInverse`、`DeferredPathSolve` 异常跳出逐 env 调用再汇总请求的机制，可以迁移为显式 `PlanningRequest`，但求解后端和请求发生时机必须保持不变。不能在第一轮重构中顺便替换已经验证的精确 IK/路径 worker。

### 6.4 阶段入口和规划等待

阶段入口产生的目标和路径存入该阶段的 `StageContext`。入口初始化具有 `UNINITIALIZED -> REQUESTED -> READY/FAILED` 状态，但 `REQUESTED` 不是新的业务阶段，也不出现在采集标签中。

当前稀疏规划在控制周期内同步批量完成，期间不执行 physics。因此重构后同样必须在目标 `READY` 后才发布该阶段第一条命令，不能为了异步化而多插入一帧保持命令。

### 6.5 阶段时钟

当前 `stage_steps` 同时承担路径索引、插值比例、最短等待和超时，而且 post-physics 转移后会在同一周期末把新阶段计数加到 1。直接“清理”这个细节会改变 RETURN_HANDLE、RESTORE_GRIPPER 等阶段的第一条命令。

新执行器最终应分别维护：

- `commands_emitted`：本阶段已发布命令数；
- `observations_evaluated`：本阶段已验收的 post-physics 样本数；
- `elapsed_steps`：用于预算；
- `path_index`：由路径执行器独立维护。

迁移时先通过特征测试记录每个阶段边界的旧值，并配置兼容映射，使发布的 waypoint、插值比例和超时帧与旧实现一致。是否进一步统一时钟语义属于单独的行为修改，不属于本次结构重构。

## 7. AutoOpen 各阶段如何组合标准功能

下表描述当前控制语义在新架构中的映射。重构时数值参数必须从现有实现原样迁移，不能借架构调整之名修改行为。

| 阶段 | 目标/入口规划 | 底盘执行器 | 机械臂执行器 | 夹爪执行器 | 验收组合 |
|---|---|---|---|---|---|
| `APPROACH` | `PoseInFrame(DOOR_FRAME, approach_offset)`，应用既有目标随机化 | `TrackBasePose` | `HoldJointPose(entry_arm_joints)` | `HoldGripperPosition(closed)` | `StableFor(AllOf(PoseWithinTolerance(base), base speeds below limits))` |
| `PREGRASP` | `AngleSweepCandidates(0/5/10/15/20 deg)`；`SelectReachableCandidate`；`PlanCartesianPath` | `HoldBasePose(previous_goal)` | `FollowCartesianPath` | `HoldGripperPosition(closed)` | `StableFor(AllOf(PathComplete, PoseWithinTolerance(EE), arm/base stable, NoAbnormalContact))` |
| `OPEN_GRIPPER` | 复用 PREGRASP 的 `PoseGoal` | `HoldBasePose` | `HoldCartesianPose(pregrasp_goal)` | `TrackGripperPosition(open)` | `StableFor(AllOf(pregrasp pose valid, ScalarWithinTolerance(gripper), gripper speed low))` |
| `CENTER_HANDLE` | `PoseFromObjectGeometry(handle, pocket/channel alignment + axial margins)`；`PlanCartesianPath` | `HoldBasePose` | `FollowCartesianPath` | `HoldGripperPosition(open)` | `StableFor(AllOf(pocket 法向/轴向误差、channel 高度误差、两端轴向余量、EE stable))` |
| `GRASP_HANDLE` | 复用 CENTER_HANDLE 目标，并通过 `BoundedAxisCorrection` 做现有垂直平衡 | `HoldBasePose` | `TrackCartesianPose(corrected capture goal)` | `HoldGripperPosition(closed=0 rad)` | `StableFor(AllOf(OpposedContactsPresent, contact force qualified, grasp geometry retained))` |
| `TURN_HANDLE` | `PlanAxisRotationPath(handle axle, handle axis, fixed endpoint)`，保留现有滚动重规划规则 | `HoldBasePose` | `FollowCartesianPath` | `HoldGripperPosition(closed)` | `AllOf(latch released, door motion established, grasp retained, StableFor(hinge speed below limit))` |
| `RETURN_HANDLE` | 线性进度生成目标把手角，`PoseOnArticulationOrbit` 逐帧转换为 EE 目标 | `HoldBasePose` | `TrackCartesianPose` | `HoldGripperPosition(closed)` | `StableFor(ObjectJointWithinTolerance(handle horizontal))` |
| `OPEN_AND_CLEAR` | `ComposePoseOffset` 构造释放终点；规划机械臂和夹爪同步路径 | `HoldBasePose` | `FollowCartesianPath` | `FollowGripperPath(closed -> open)` | `StableFor(AllOf(gripper open, ContactAbsent(handle)))` |
| `RETREAT_ARM` | `ComposePoseOffset(current, door normal retreat + vertical offset)`；`PlanCartesianPath` | `HoldBasePose` | `FollowCartesianPath` | `HoldGripperPosition(open)` | `StableFor(AllOf(PathComplete, PoseWithinTolerance(EE), ContactAbsent(handle)))` |
| `RESTORE_GRIPPER` | `RandomBallPoseCandidates(retract center)`；`SelectReachableCandidate`；`PlanCartesianPath` | `HoldBasePose` | `FollowCartesianPath` | `FollowGripperPath(open -> closed)` | `StableFor(AllOf(PoseWithinTolerance(sampled intent target), arm stable, gripper closed))` |
| `ALIGN_PANEL` | `PoseFromObjectGeometry(panel station)` | `TrackBasePose` | `HoldJointPose(entry joints)` | `HoldGripperPosition(closed)` | `StableFor(AllOf(PoseWithinTolerance(base), base speeds below limits))` |
| `CONTACT_PANEL` | `PlanarGridCandidates(panel baseline)`；`SelectReachableCandidate` 一次批量 IK；生成门外 30 mm standoff `PoseGoal` | `HoldBasePose` | `TrackCartesianPose` | `HoldGripperPosition(closed)` | `StableFor(AllOf(finger clearance、panel tangent error、axis error、EE speed))` |
| `PUSH_OPEN` | `ProgressLeadController` 生成目标门角，`PoseOnArticulationOrbit` 转成动态 EE 目标，再组合 `ContactRetreatCorrection` | `HoldBasePose` | `TrackCartesianPose` | `HoldGripperPosition(closed)` | `AnyOf(door angle reached, arm reposition condition)`；超速/超力/过冲是独立保护条件 |
| `PUSH_ARM_RETRACT` | 每次入口重新 `RandomBallPoseCandidates`；`SelectReachableCandidate`；入口保持 5 帧后以每步 4 mm 跟踪目标 | `HoldBasePose` | `TrackCartesianPose` | `HoldGripperPosition(closed)` | `StableFor(AllOf(PoseWithinTolerance(sampled intent target), arm speed low))` |
| `BASE_ADVANCE` | `PoseSatisfyingConstraints(recontact reachable, shoulder safe, doorway center)`；`MonotonicGoalFilter(forward)`，逐帧更新 | `TrackBasePose` | `HoldJointPose(entry joints)` | `HoldGripperPosition(closed)` | `StableFor(AllOf(recontact reachable, PoseWithinTolerance(base), base speeds below limits))` |
| `THROUGH` | `PoseSatisfyingConstraints(tail plane distance=1 m, doorway center, target yaw)` | `TrackBasePose` | `TrackJointPose(middle preset)` | `HoldGripperPosition(closed)` | 入口稳定条件后，`StableFor(AllOf(tail clearance、base pose、base/arm/gripper/door speed、arm preset、upright、NoAbnormalContact))` |
| `SUCCESS` | 无新目标 | `HoldBasePose` 或 `StopBase`，按当前语义固定 | 明确保持最终目标 | 明确保持最终目标 | 终态 |
| `FAILURE` | 无新目标 | 明确安全命令 | 明确安全命令 | 明确安全命令 | 终态 |

这张表是主流程最终应表达的内容。实现不能把表中一整行重新塞进一个阶段专用函数。

### 7.1 当前转移相位必须保留

| 转移 | 相位 | 原因 |
|---|---|---|
| `APPROACH -> PREGRASP` 至 `CONTACT_PANEL -> PUSH_OPEN` 的常规到达链路 | `POST_PHYSICS` | 使用刚执行完 physics 的实测状态验收 |
| `PUSH_OPEN -> PUSH_ARM_RETRACT` | `PRE_COMMAND` | 超出舒适区后当前周期立即改发回撤命令 |
| `PUSH_ARM_RETRACT -> BASE_ADVANCE` | `PRE_COMMAND` | 回撤稳定后当前周期切换为关节保持与底盘移动 |
| `BASE_ADVANCE -> CONTACT_PANEL` | `PRE_COMMAND` | 再接触条件稳定后立即锁存新站位并准备门板目标 |
| `CONTACT_PANEL/PUSH_OPEN/PUSH_ARM_RETRACT/BASE_ADVANCE -> THROUGH` | `POST_PHYSICS` | 根据最新门角和门速判断 |
| 任意阶段安全失败 | 与对应测量相同的相位 | 不允许多执行一帧已知危险命令 |

阶段定义中必须直接显示 `phase=`，不能让通用执行器根据阶段名称硬编码。

## 8. 主任务图应有的形态

主任务文件需要直接展示动作、目标、验收和转移。以下仅表示接口形态，参数值仍以现有代码为准：

```python
AUTO_OPEN = TaskGraph(
    stages={
        Stage.APPROACH: StageSpec(
            goals={
                "base": PoseInFrame(
                    frame=DOOR_FRAME,
                    pose=APPROACH_POSE,
                    randomization=APPROACH_NOISE,
                    update=ON_ENTRY,
                ),
            },
            actions=(
                TrackBasePose(goal="base", limits=BASE_LIMITS),
                HoldJointPose(goal=ENTRY_ARM_JOINTS),
                HoldGripperPosition(goal=CLOSED),
            ),
            done=StableFor(
                AllOf(
                    PoseWithinTolerance(BASE, goal="base"),
                    LinearSpeedBelow(BASE, APPROACH_LINEAR_SPEED),
                    AngularSpeedBelow(BASE, APPROACH_ANGULAR_SPEED),
                ),
                frames=APPROACH_STABLE_FRAMES,
            ),
            budget_steps=1000,
            transitions=(
                OnDone(Stage.PREGRASP, phase=POST_PHYSICS),
                OnTimeout(Stage.FAILURE, phase=POST_PHYSICS),
            ),
        ),
        Stage.THROUGH: StageSpec(
            goals={
                "base": PoseSatisfyingConstraints(
                    frame=DOOR_FRAME,
                    constraints=(
                        TailBeyondPlane(distance=1.0),
                        LateralAtDoorwayCenter(),
                        YawEquals(THROUGH_YAW),
                    ),
                    update=ON_ENTRY,
                ),
                "arm": JointPose(MIDDLE_PRESET),
            },
            actions=(
                TrackBasePose(goal="base", limits=BASE_LIMITS),
                TrackJointPose(goal="arm", limits=ARM_JOINT_LIMITS),
                HoldGripperPosition(goal=CLOSED),
            ),
            done=StableFor(
                AllOf(
                    PoseWithinTolerance(BASE, goal="base"),
                    JointPoseWithinTolerance(ARM, goal="arm"),
                    SystemSpeedsBelow(THROUGH_TERMINAL_SPEEDS),
                ),
                frames=TERMINAL_STABILITY_FRAMES,
            ),
            budget_steps=1100,
            transitions=(
                OnDone(Stage.SUCCESS, phase=POST_PHYSICS),
                OnTimeout(Stage.FAILURE, phase=POST_PHYSICS),
            ),
        ),
    }
)
```

这里 APPROACH 与 THROUGH 使用的是同一个 `TrackBasePose` 和同一个 `PoseWithinTolerance`。门框站位和尾部净空只通过目标约束表达。

PUSH_OPEN 的关键部分应显式写成通用组件组合，而不是一个阶段专用 action：

```python
Stage.PUSH_OPEN: StageSpec(
    goals={
        "arm": ModifiedGoal(
            source=PoseOnArticulationOrbit(
                joint=DOOR_HINGE,
                pivot=HINGE_FRAME,
                tracked_point=SELECTED_PANEL_CONTACT,
                progress=ProgressLeadController(
                    target_position=DOOR_OPEN_ANGLE,
                    target_velocity=PUSH_TARGET_SPEED,
                    kp=PUSH_SPEED_KP,
                    ki=PUSH_SPEED_KI,
                    max_lead=PUSH_MAX_LEAD,
                ),
                update=EVERY_STEP,
            ),
            modifiers=(
                ContactRetreatCorrection(
                    metric=PANEL_CONTACT,
                    max_force=PUSH_FORCE_LIMIT,
                    max_approach_speed=PUSH_APPROACH_SPEED_LIMIT,
                ),
            ),
        ),
    },
    actions=(
        HoldBasePose(goal=LAST_BASE_MOTION_GOAL),
        TrackCartesianPose(goal="arm", limits=PUSH_EE_LIMITS),
        HoldGripperPosition(goal=CLOSED),
    ),
    transitions=(
        On(
            ArmReachAbove(COMFORT_RADIUS),
            Stage.PUSH_ARM_RETRACT,
            phase=PRE_COMMAND,
        ),
        On(
            DoorAngleAbove(DOOR_OPEN_ANGLE),
            Stage.THROUGH,
            phase=POST_PHYSICS,
        ),
    ),
)
```

这里 `ProgressLeadController`、`PoseOnArticulationOrbit`、`ContactRetreatCorrection` 和 `TrackCartesianPose` 都不依赖 AutoOpen 阶段枚举，可原样用于 AutoClose；AutoClose 只需改变目标门角、速度方向、接触点和阶段图。

## 9. 阶段入口事件和跨阶段数据

部分阶段需要在转移时生成下一阶段目标。不能继续把这些计算写进前一阶段的 `advance()` 分支。统一执行器应按以下顺序处理一组发生转移的 env：

1. 验收当前阶段；
2. 根据显式转移条件选择下一阶段；
3. 退出当前阶段；
4. 初始化下一阶段的目标提供器并生成 `PlanningRequest`；
5. coordinator 与其他 env 的同类请求合并求解；
6. 将生成的目标与路径写入下一阶段 `StageContext`；
7. 如果转移发生在 `PRE_COMMAND`，当前周期直接生成新阶段命令；如果发生在 `POST_PHYSICS`，下一周期生成新阶段命令。

需要跨阶段保留的数据通过有名称的 `TaskMemory` 键传递，例如：

- `selected_grasp_pose`；
- `selected_panel_contact_pose`；
- `last_base_motion_goal`；
- `handle_turn_endpoint`；
- `active_retract_pose`。

不得通过几十个可空成员变量暗中表达阶段是否已经准备完成。

## 10. 建议代码边界

最终目录边界建议如下。最上层脚本必须保留阶段枚举和完整任务图，不能再次变成只有若干行的启动包装器：

```text
gr00t/rl/task_primitives/
  types.py                 # Snapshot、Goal、Command、Result
  behaviors/
    base.py                # TrackBasePose / HoldBasePose / StopBase
    arm.py                 # Cartesian、path、joint behaviors
    gripper.py             # gripper behaviors
    modifiers.py           # 通用目标修正器
  goals/
    frames.py              # frame-relative and latched goals
    constraints.py         # constraint-composed goals
    randomization.py       # target-level randomization
  planning/
    cartesian.py           # IK、Cartesian path
    joints.py              # joint path
    candidates.py          # batched candidate selection
  conditions/
    pose.py
    motion.py
    contact.py
    combinators.py
  adapters/
    b2z1.py                # 物理语义命令到 B2Z1 低层 buffer
  execution.py             # masked batched stage executor

gr00t/rl/auto_open/
  geometry.py              # 门、把手、门框领域几何模型
  goals.py                 # AutoOpen 使用的目标参数和约束组合
  config.py                # 阈值、预算、随机化配置
  diagnostics.py           # AutoOpen 诊断字段

gr00t/rl/scripts/
  doorman_auto_open.py     # Stage、完整声明式阶段图、CLI main
  run_auto_open.sh         # 唯一正式启动入口
```

通用模块不能 import AutoOpen 的 `Stage` 枚举，也不能出现 `approach`、`through`、`push_open` 等阶段名称。AutoOpen 可以 import 通用模块。

`doorman_auto_open.py` 可以声明任务特有的阶段组合、参数和转移，但不实现底层控制算法。现有唯一启动入口 `gr00t/rl/scripts/run_auto_open.sh` 保持不变。

## 11. 哪些逻辑不应错误地通用化

以下内容是 AutoOpen 的领域配置或任务图，不要求伪装成全任务通用算法：

- 门把手 pocket/channel 的几何定义；
- 把手轴座侧和自由端余量；
- 门板推门基准点的数值；
- 门铰链开门角度；
- AutoOpen 的阶段顺序和回环关系；
- 各阶段预算、容差和稳定帧数。

它们可以使用通用目标构造器和条件，但参数本身应留在 AutoOpen 任务目录中，清楚可见。

## 12. 基于当前代码的可行性审计

本节不是概念推演，而是对当前代码的实现级映射。结论是方案可实施，但必须分阶段迁移，不能直接把 `DoorController` 替换成一套全新框架。

### 12.1 已有实现与目标模块的对应关系

| 当前实现 | 已验证职责 | 目标位置 | 迁移方式 |
|---|---|---|---|
| `_base_commands_exact` | 批量底盘位姿 PD、速度上下限和误差计算 | `TrackBasePose` 数值内核 | 原函数先原样搬迁并做逐元素等价测试，再 tensor 化 |
| `_limit_base_commands`、`_stabilize_base_command`、`_zero_cross_base_reversal` | 全局限速和反向抑制 | `TrackBasePose` 的可配置 command filters | 将内部状态改成 `[num_envs, ...]` tensor，不改阈值或更新顺序 |
| `_base_hold_target`、`_set_base_tracking_target` | 跨阶段保持上一个移动目标 | `LatchPreviousGoal` + `HoldBasePose` | 将四个散落字段合并成一个 `PoseGoal` |
| `_slew_ee_target`、`_slew_orientation` | Cartesian 位置和姿态连续限速 | `TrackCartesianPose` 的 rate limiter | 使用 batched command state，保持当前每阶段 max step |
| arm joint/EE mode 分支及 `set_teleop_targets` | 控制模式和 buffer 编码 | `ArmCommand` + `B2Z1CommandAdapter` | 先集中编码，不改变低层接口 |
| `_arm_pose_is_stable`、`_BatchedStabilityTracker` | EE 实测窗口稳定性 | `StableFor(PoseWindowStable(...))` | 保留 GPU 计算和边界精确回退，移除双份状态 |
| `_geometry_metrics`、`_batched_geometry_metrics_gpu/numpy` | 把手和夹爪语义几何 | `HandleCaptureMetrics` | 先包成统一 batched result，保留临界阈值回退 |
| `_panel_finger_clearance`、GPU batch 版本 | 指尖与门板距离 | `PanelContactMetrics` | 保留现有 GPU backend |
| `_plan_pregrasp_path`、`_build_steady_compensated_path` | 预抓取候选和精确路径 | `SelectReachableCandidate` + `PlanCartesianPath` | 请求参数类型化；第一阶段继续调用相同 worker |
| `_build_release_inverse_path` | 释放、回撤、恢复路径 | `PlanCartesianPath` + 同步 gripper path | 合并重复入口参数，不改 waypoint |
| `_sample_reachable_arm_retract_pose` | 随机采样、批量 IK、首个可达候选选择 | `ApplyTargetRandomization` + `SelectReachableCandidate` | 保留 seed、候选顺序和 rejection metadata |
| `_prepare_panel_contact_command`、`_select_reachable_panel_contact_target` | 门板网格候选与单次批量 IK | `PlanarGridCandidates` + `SelectReachableCandidate` | 保留同一候选顺序和代价排序 |
| `_base_advance_target` | 中线上的可达/肩部安全目标及单调约束 | `PoseSatisfyingConstraints` + `MonotonicGoalFilter` | 先拆成纯函数，随后批量 tensor 化 |
| `_turn_path_target`、RETURN_HANDLE 的轴旋转公式 | 绕关节轴的 EE 位姿 | `PoseOnArticulationOrbit` / `PlanAxisRotationPath` | 使用同一轴、轴心、符号和 quaternion 顺序做数值对比 |
| PUSH_OPEN 的 PI 提前量和接触退让分支 | 目标门角、轨道 EE 目标和安全修正 | `ProgressLeadController` + `PoseOnArticulationOrbit` + modifiers | 按计算顺序拆开，输出逐字段等价后才替换 |
| `advance()` 中的计数器和布尔组合 | 稳定验收、保护和转移 | conditions + transition executor | 每个旧分支建立真值表和边界回放测试 |
| `DeferredSteadyInverse`、`DeferredPathSolve` 汇总逻辑 | 跨 env 稀疏请求批处理 | `PlanningRequestCoordinator` | 先改成显式请求，不更换 worker/backend |
| `_BatchedRunnerView`、逐 env controller loop | 当前兼容层 | 最终删除 | 仅在所有命令和验收都已 mask-batch 化后删除 |

这张表说明主要数值能力已经存在。重构的核心不是重新发明控制算法，而是把其输入、输出和状态所有权标准化，再逐步把逐 env 调度替换为 mask-batch 调度。

### 12.2 六个实际难点及确定处理方式

1. **当前 FSM 仍是 CPU 逐 env。**

   处理：行为和条件状态从 Python 对象字段迁移成定长 batched tensor；变长路径使用 padded tensor、length 和 index；诊断字符串只在 episode 结束或失败时从 code 映射生成。逐帧 GPU 路径不维护 Python 字典列表。

2. **路径长度和阶段不同，环境会异步分叉。**

   处理：每个 stage 使用 `stage_mask = stage_id == id`；路径保存为 `[N, max_waypoints, ...]`，每个 env 有独立 `path_length/path_index`。同一周期可以同时执行多个 stage 的标准行为，结果通过 mask 写入唯一 command buffer。

3. **精确 IK/路径 worker 当前在 CPU，且请求稀疏。**

   处理：它不是逐帧控制接口的一部分。`PlanningRequestCoordinator` 只收集新进入相关阶段的 mask，按 request kind 批量调用现有 worker。最终是否替换为 GPU 求解器是独立性能任务，不阻塞结构重构。

4. **几何 GPU 结果在阈值附近有精确回退。**

   处理：`MetricResult` 同时带 value 和 confidence/boundary mask。condition 先使用 GPU batch，对 boundary mask 批量调用现有 NumPy/精确后端。不能为了接口整齐删除回退而改变成功率。

5. **命令生成会修改状态，甚至触发阶段转移。**

   处理：将其拆成 `PRE_COMMAND` condition、纯目标更新器和 behavior state update；transition 只能由 executor 提交。通过保存的输入快照比较“旧 command 调用后的状态”与“新三步组合后的状态”。

6. **控制命令与数据监督目前通过同一 high action 间接关联。**

   处理：`TaskCommand` 成为意图和模式语义的唯一源，`B2Z1CommandAdapter` 生成低层 buffer；physics 后从低层状态生成 `DispatchedCommandSnapshot`。数据 action 使用实际 dispatched 数值，并直接采用 `TaskCommand` 声明的 active/inactive/reset 语义。增加断言，确保两者对应，避免采集器再次按数值猜测上层意图。

### 12.3 内存规模和 tensor 表达可行性

所有逐帧控制状态都是固定小张量：stage id、计数器、位姿、滤波器状态和少量接触历史。即使 10000 env，也远小于仿真状态本身。变长数据只有 Cartesian 路径；当前路径通常几十个 waypoint，可按阶段只为活跃请求分配 padded batch，完成后释放或复用池，不需要为每个 env 永久保存所有阶段路径。

阶段图本身是 Python 静态配置，只在初始化时解析；逐帧执行使用编译后的 stage id、mask、参数 tensor 和函数表，不逐 env 解释 `StageSpec`。

### 12.4 方案不依赖尚不存在的能力

- 底盘批量 PD 已有 `_base_commands_exact`；
- 把手几何和门板间隙已有 GPU batch 实现；
- 稳定窗口已有 GPU tracker；
- 候选 IK 和路径已有稀疏批量 worker；
- 低层本来就接受整批 base、EE、joint-mode 和 gripper buffer；
- 当前 runtime 已经能按 env mask 选择几何和规划请求。

因此不存在必须先解决未知 IK、改造物理引擎或重写低层策略才能落地的前置条件。最大工作量是 FSM 状态张量化和严格的阶段边界等价验证，而不是算法不可行。

### 12.5 dtype、随机数与 seed 一致性

现有底盘计算有意混用 float32 状态和 float64 yaw/trigonometry，以对齐标量历史行为；IK 使用 float64；低层 command buffer 最终为 float32。每个新 primitive 必须声明 compute dtype 和 output dtype，不能统一“顺手 cast 成 float32”。condition 在阈值附近继续使用现有精确回退，保证分支选择一致。

随机化保留两种现有确定性键：

- 语义目标噪声由 `(episode seed, semantic key)` 唯一确定；
- 收回目标由 `(episode seed, retract stream, entry index)` 唯一确定。

在结构重构期间继续使用当前 NumPy `SeedSequence/default_rng` 算法，因此同 seed、同入口次数必须采到相同值。它们只在阶段入口的稀疏规划服务中运行，不进入逐帧 FSM。若未来要把该稀疏 RNG 也迁到 GPU，必须单独验证随机分布并重新建立基线，不能混入本次等价重构。

相机开关、GPU/rank、batch size 和 env slot 不得进入上述随机键。入口次数是每个 episode 独立的 tensor 计数，不得使用全局调用顺序。

## 13. 可执行迁移顺序

每一步只替换一种能力；未迁移部分继续由旧控制器执行，但同一能力在正式路径中只能有一个权威实现。兼容层只负责数据格式，不允许维护第二套语义。

### 13.1 建立特征测试，不改运行行为

1. 保存各阶段稳定段、入口前一帧、入口后一帧、路径末帧和失败边界的状态快照。
2. 记录旧实现的目标记录、三个通道命令、内部计数器更新、验收结果和转移。
3. 固定 float dtype、quaternion 顺序、坐标系和随机数序列。
4. 为 `PRE_COMMAND` 与 `POST_PHYSICS` 分别建立回放入口。

### 13.2 集中类型和命令适配

引入 `TaskSnapshot`、goal types、`TaskCommand`、`B2Z1CommandAdapter` 和 `DispatchedCommandSnapshot`。旧 `command()` 仍生成原值，但先写入 `TaskCommand`，再由单一 adapter 写低层 buffer；采集器继续记录低层实际 dispatched 数值，但监督模式改为读取显式语义。验收标准是低层 buffer、数据 action 和模式标志逐元素一致。

### 13.3 抽取底盘标准执行器

先让 APPROACH 使用 `TrackBasePose`，再依次接入 ALIGN_PANEL、BASE_ADVANCE、THROUGH 和所有 `HoldBasePose` 阶段。每接入一种参数组合，都用相同 snapshot 比较旧/新 command 与 filter state。完成后删除 `_command_approach` 和阶段内重复底盘分支。

### 13.4 抽取机械臂、夹爪执行器和路径时钟

依次抽取 joint hold、Cartesian hold、Cartesian track、Cartesian path、gripper position/path。保留当前 waypoint、rate limit、arm mode 和插值首帧。路径执行器独立拥有 index，阶段预算不再兼任路径索引，但通过兼容映射保持输出序列。

### 13.5 抽取测量器和验收条件

先迁移无状态阈值，再迁移 `StableFor`、接触时效窗口、超速连续帧和 no-progress 条件。每个旧 stage 分支建立真值表，确认相同 snapshot/history 得到相同 boolean、计数和失败原因。

### 13.6 显式化规划请求和阶段上下文

将异常式 deferred planning 替换为 `PlanningRequest`；仍调用当前 worker。目标、路径和 provenance 进入 `StageContext/TaskMemory`，逐项删除 `DoorController` 中对应可空字段。

### 13.7 编译阶段图并替换巨型分支

当三个通道、验收和规划都已有标准模块后，启用统一 executor。先以 batch size 1 回放全部阶段，再比较多 env 输出。此时才能删除 `command()`/`advance()` 巨型分支。

### 13.8 张量化 FSM 并删除逐 env 兼容层

把 stage id、计数器、behavior state 和 condition state 转为设备 tensor；按 stage mask 调标准模块。删除每 env `DoorController`、`_BatchedEnvView`、完整状态每帧 CPU copy 和逐 env command/advance loop。稀疏 CPU planner 只在阶段边界接收必要的小张量。

### 13.9 闭环验收

先运行四 seed 带视频回归，再运行固定入口的 1000 seed。只有模块回放、阶段时序、成功率、失败分布和耗时均通过，才替换当前基线。

## 14. 行为等价要求

本次重构默认只改变代码组织，不改变行为。任何行为调整必须另立问题、另做 A/B，并获得明确认可。

实施时应按以下顺序验证：

1. **纯函数特征测试**：用相同输入快照比较旧函数和新模块输出。
2. **逐模块等价**：目标、命令、验收布尔值和诊断值分别比较，不能只比较最终成功率。
3. **阶段边界回放**：覆盖入口首帧、路径末帧、稳定窗口和超时边界。
4. **单环境/多环境一致性**：同 seed 在 batch size 1 与任意 batch slot 中输出一致。
5. **四 seed 视频回归**：确认阶段时序和实际动作一致。
6. **1000 seed 回归**：相对当前 `916/1000` 基线检查成功率、失败阶段分布和耗时。

局部抽取不需要每次都运行完整闭环；优先使用保存的相同状态输入直接比较模块输出。只有完成一组有闭环影响的模块替换后才运行对应回归。

## 15. 重构完成标准

只有同时满足以下条件，才算完成本次架构重构：

1. 主任务文件完整、直接地列出所有阶段的三个控制通道、目标、验收、预算和转移。
2. APPROACH、ALIGN_PANEL、BASE_ADVANCE、THROUGH 使用同一个 `TrackBasePose` 实现。
3. 所有保持站位阶段通过 `HoldBasePose` 复用同一跟踪内核。
4. 所有 Cartesian 单点和路径动作分别统一到标准执行器。
5. 所有夹爪动作由位置目标或路径明确表达，反馈不反写为监督命令。
6. 目标生成、命令生成和验收共享同一类型化目标记录，名义目标、采样意图、低层命令和验收参考均可追溯。
7. `command()`/`advance()` 的阶段巨型分支被统一执行器取代。
8. 通用模块中不存在 AutoOpen 阶段名或一阶段一函数的伪抽象。
9. 单环境与多环境仍为同一批量实现，逐帧主路径无逐 env CPU 循环。
10. 唯一正式运行入口不变，数据采集继续作为控制流程的被动观察者。
11. 现有测试、四 seed 回归和 1000 seed 基线完成等价性验收。

在以上设计经过确认前，不开始控制代码重构。
