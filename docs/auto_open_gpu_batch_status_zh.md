# AutoOpen 当前版本说明

本文只维护当前推门/拉门的统一入口、控制规则、最近完整基线和已知缺口。实验过程与对照记录见[验收台账](door_baseline_validation_progress_zh.md)，不在本文堆叠版本修改记录。

## 当前基线

2026-11-03，正式canonical入口，四卡各250个环境，全部1000个请求seed计入分母；无图像、无视频、无补图，保留完整物理与严格数据验收。

| 模式 | Seed闭区间 | 严格通过 | 严格拒收 | FSM流程完成 | 入口总耗时 |
| --- | --- | ---: | ---: | ---: | ---: |
| 推门 | 1700–2699 | 871/1000，87.1% | 129 | 969/1000 | 2166秒，36分06秒 |
| 拉门 | 1700–2699 | 799/1000，79.9% | 201 | 945/1000 | 3021秒，50分21秒 |

成功率指严格数据通过率，不是门打开或流程到达SUCCESS的比例。异常接触、穿透、指令质量或其他验收失败均拒收，不补采后改变分母。

与本轮起始参考81.3%/81.8%、2707/2980秒相比：推门提高5.8个百分点并减少541秒；拉门降低1.9个百分点、增加41秒。按用户允许约80%的可比口径收口；拉门准确数值是79.9%，不是80.0%。没有显著耗时回退。

配置边界：本轮debug、上述canonical及配套远程采集使用`TRUNK_CONTROL_FILTER_ENABLED=True`。依照此前要求，Git提交中的该开关默认仍为False，本地工作区保持True；远程使用声明且校验哈希的True配置覆盖。关闭滤波的默认运行不能冒充本文基线。启动前应确认实际开关。

标准debug推门/拉门各1700–1703均严格4/4，共8/8；全部六视角视频已替换到[Debug报告](lab30_door_debug_results_zh.md)及[HTML视频页](lab30_door_debug_results_zh.html)。典型穿门失败1704–1707也严格4/4。小规模不能替代上述两个1000seed。

## 严格失败

下表按每条数据的首个拒收原因互斥分类，每列合计等于严格拒收数。后续问题不重复计数。

| 首要拒收原因 | 推门 | 拉门 |
| --- | ---: | ---: |
| 异常环境接触 | 102 | 130 |
| 推板门速连续3帧超过30度/秒 | 0 | 30 |
| 机械臂目标短时往返 | 0 | 15 |
| GRASP_HANDLE握把超时 | 10 | 9 |
| APPROACH_HANDLE到位/稳定超时 | 5 | 7 |
| MOVE_TO_PANEL_STANDOFF逆映射规划失败 | 5 | 1 |
| 接触穿透超限 | 0 | 4 |
| 初始开缝q5目标不连续 | 2 | 2 |
| 底盘指令短时往返 | 2 | 0 |
| 预抓取路径规划失败 | 2 | 1 |
| 预抓取无可达pitch | 0 | 1 |
| OPEN_DOOR_WITH_HANDLE初始开缝超时 | 0 | 1 |
| INSERT_AROUND_HANDLE插入把手超时 | 1 | 0 |
| 合计 | 129 | 201 |

异常接触进一步按首次异常事件阶段拆分，每条数据计一次，不是接触帧数。

| 首次异常阶段 | 推门 | 拉门 |
| --- | ---: | ---: |
| APPROACH_HANDLE：靠近把手 | 70 | 70 |
| WITHDRAW_FROM_HANDLE：松手后回撤 | 9 | 25 |
| MOVE_TO_PANEL_STANDOFF：前往推板目标 | 10 | 3 |
| MOVE_TO_PANEL_STATION：首次推板站位 | 7 | 0 |
| PASS_THROUGH_DOORWAY：穿门 | 3 | 23 |
| MOVE_TO_PREGRASP：预抓取就位 | 2 | 3 |
| PUSH_PANEL_OPEN：推门板 | 1 | 5 |
| FOLLOW_WALL_TO_PANEL：贴门板移动 | 0 | 1 |
| 合计 | 102 | 130 |

## 主要缺口

- 早期接触是剩余主要损失：两种模式各70条APPROACH异常，其中68条夹爪/腕部碰把手、2条碰门框；不是推板阶段撞门。现行规则只按初始夹爪高度区间和前伸量增大靠近距离，不能保证行走动态全程净空。本轮没有继续修改该规则。
- 回撤异常推门9条、拉门25条，首事件均为腕部碰把手。释放高度恢复与开爪验收通过，不等于整条后撤扫掠绝对无接触；没有放宽接触豁免或将它们算成功。
- 拉门穿门首要异常23条：机械臂碰门框19条，腿碰门板4条。19条中13条发生在底盘零指令的折臂过程，6条发生在实际侧移/转向；逐关节独立限速不等于关节协调直线插值，终点紧凑不能证明途中也紧凑。相关实际状态分析在`artifacts/door_baseline_validation/pull_remaining_frame_analysis.json`。仍严格拒收，不宣称穿门碰撞全部解决。
- 拉门推板超速仍有30条作为首因、33条实际运行失败：门速连续3帧超过30度/秒。尖端4度超前是位置目标上限，不是门速硬约束；反馈减小超前也不等于立即制动门板惯性。本轮未修改这项控制或阈值。
- 机械臂目标短时往返15条继续按指令质量拒收，未放松窗口或阈值；未在本轮完成独立因果归因。
- 推板规划没有找到满足联合检查的候选，不等于完整六自由度空间绝对无解。推门实际规划失败11条、首因5条；拉门实际2条、首因1条。不能把首因表与运行失败相加。

本次自主修复只作用拉门THROUGH：穿门前折臂并保持紧凑，机尾清空后恢复middle。最初同版拉门46.5%中的穿门首要异常由360降至23；净新增334条严格通过（336条新通过，2条原通过转为腿碰板拒收）。没有新增FSM运行超时，流程完成数仍945/1000。推门修复前后的1000条逐episode验收报告完全一致，不只是通过率相同。备用更小q2姿态仅离线验证，没有混入本基线。

## 统一入口

分支`auto-open`，唯一入口`gr00t/rl/scripts/run_auto_open.sh`；`DOOR_MOTION=push|pull`选择任务方向，与采集模式独立。

| AUTO_OPEN_MODE | 行为 |
| --- | --- |
| official，默认 | 无相机完整采集、严格筛选，通过轨迹补双鱼眼，再严格复验并正式发布 |
| canonical | 无相机完整采集及严格验收，不进入图像链路 |
| debug | 六相机完整采集及严格验收，默认输出六视角拼接视频，不正式发布 |

```bash
AUTO_OPEN_MODE=canonical DOOR_MOTION=push GPU_IDS=0,1,2,3 \
SEED_START=1700 SEED_END=2699 bash gr00t/rl/scripts/run_auto_open.sh
```

拉门改`DOOR_MOTION=pull`；四seed debug改`AUTO_OPEN_MODE=debug SEED_END=1703`。每卡一个Isaac Sim进程、进程内多个独立环境；推门预算4000步，拉门16000步，另有各阶段预算。控制50Hz、物理200Hz。

结果位于`datasets/auto_open/{debug,canonical,releases,work}`。正式批次名包含seed区间、实际发布数、相机类型和时间，不维护latest指针。official补图逐帧恢复已记录的机器人/关节/门状态，不重新运行策略或FSM。

## 标准编排

详见[标准接口](auto_open_standard_interfaces_zh.md)。目标来源为直接目标或GPU候选/路径规划；机身执行目标位姿或预计算位姿轨迹，统一进入`base_target`；机械臂使用`arm_pose_target`、`arm_trajectory`、`arm_joint_target`，夹爪使用`gripper_target`。

`stage_interfaces.py`声明阶段模式，`execution.py`是命令写入层，`TensorCommandBuffers.flush`是环境写入出口。通用机身/机械臂稳定窗口共用，阶段组合任务条件，不另建逐环境控制协议。

推、拉门共享把手段：

```text
APPROACH_HANDLE → MOVE_TO_PREGRASP → OPEN_GRIPPER → INSERT_AROUND_HANDLE
→ GRASP_HANDLE → PRESS_HANDLE → OPEN_DOOR_WITH_HANDLE → RESET_HANDLE
→ RELEASE_HANDLE → WITHDRAW_FROM_HANDLE → STOW_ARM_AND_CLOSE_GRIPPER
```

推门后半段：

```text
MOVE_TO_PANEL_STATION → MOVE_TO_PANEL_STANDOFF → PUSH_PANEL_OPEN
角度/投影不足且超出舒适区：STOW_ARM_AFTER_PUSH → ADVANCE_TO_NEXT_PUSH_STATION → MOVE_TO_PANEL_STANDOFF
投影满足：收臂 → PASS_THROUGH_DOORWAY → SUCCESS
```

拉门后半段：

```text
MOVE_TO_GAP_STATION → INSERT_INTO_DOOR_GAP → PRY_DOOR_OPEN
→ STOW_ARM_AFTER_PULL → FOLLOW_WALL_TO_PANEL → MOVE_TO_PANEL_STANDOFF → PUSH_PANEL_OPEN
投影不足：STOW_ARM_AFTER_PULL → FOLLOW_WALL_TO_PANEL → MOVE_TO_PANEL_STANDOFF
投影满足：收臂 → PASS_THROUGH_DOORWAY → SUCCESS
```

首次拨门至少30度；后续复用推板，不重复首次拨门。PASS_THROUGH_DOORWAY内部连续对准门框并前进，不另设拉门对准门框阶段。

## 主要控制参数

- APPROACH名义站位：到把手抓取中心的门法向距离0.85m；若初始夹爪与把手高度区间间隔小于2cm，按实际夹爪前伸加2cm只增不减。
- 底盘位置内/外阈值2/10cm，yaw内/外阈值0.05/0.15rad，稳定窗口25帧；各阶段保留任务附加条件。
- 底盘速度命令上限0.4m/s、0.6rad/s；非零平移至少0.15m/s，非零转动至少0.2rad/s。预计算轨迹上限0.25m/s、0.40rad/s，逐帧取点，末尾保持终点；不按跟踪误差暂停，也不在路段交界额外等待验收。
- 首次横移：先后退20cm，再前往站位，属于一条预计算轨迹。当前局部采样x=0.535–0.555m、y=0.630–0.650m、yaw=-132至-126度，随门方向镜像；32候选，最多四轮选站。插入目标需在0.6m内。
- 推门首次站位是一条先侧移对准板面中心、再前进的轨迹，只在末端验收；以首次推点搜索范围中心为参考。后续前移沿固定射线，以旧推点进入参考中心40cm水平半径为停止条件；无交集取射线最近点。前移目标的trunk原点不越过门框，停稳后重新选推点，不把40cm当作所有新推点的硬范围。
- FOLLOW_WALL_TO_PANEL：靠墙前肩先沿墙走，随后保持到门轴半径；左/右门轴半径0.40/0.60m。yaw按arcsin规则、机身宽度加20cm计算。预计算路径不依赖门角，发布点只在其夹角小于实际门角时前进，索引不后退；到位验收仍要求门板停止旋转。
- 进入THROUGH：门板含厚度的门框平面投影，与假想居中且垂直门框的狗身碰撞包络间隔至少10cm。狗身含机身/背筐/腿，不含机械臂；在线与数采共用GPU几何，不再固定89度。
- 拉门THROUGH携臂：保留入口实际q1，q2–q6目标为`[1.45,-0.8,1.1,0,0] rad`，使用现有JOINT接口；实际关节进入0.08rad范围后，从实际机身位姿计算原穿门轨迹。机尾清空1m后单向恢复middle，避免在尾部迟滞边界反复切换目标。
- THROUGH终态：机尾越框1m，恢复middle并满足机械臂、夹爪、机身和门的稳定条件及原50帧数据保持要求。
- RELEASE_HANDLE恢复闭爪前记录的实际世界系EE高度，允许误差6mm；WITHDRAW_FROM_HANDLE仅沿原门板法向退10cm，不额外抬高2cm。

## 机械臂规划

低层wholebody接收height-invariant EE指令，不是所有阶段都直接下发关节角；预设关节模式使用独立标准JOINT接口。轨迹入口批量逆映射，执行发布缓存参考，低层跟踪。

MOVE_TO_PREGRASP使用预计算避把手路径并批量验证pitch与路径。MOVE_TO_PANEL_STANDOFF推拉共用连续区域生成：门板接触线与0.70m平面工作空间相交，保留板边、距门轴至少20cm、把手避让及关节/精度/整臂净空检查。每yaw取10个径向位置，7个yaw覆盖±30度；拉门9个高度0.80–1.20m，共630候选；推门11个高度0.80–1.30m，共770候选。

拉门保留至少5度可用推弧要求。推门优先可用推弧，不以至少5度拒绝全部短弧；验证连续到位路径和站高裕量，必要时使用先旋转第一关节再展开的过渡备选。首次拉门半径外对准使用单点控制，固定方向前伸使用连续轨迹。

PUSH_PANEL_OPEN超前量定义在指定夹爪尖端，动态上限4度，门速反馈减小超前；EE每周期最多变化3mm。不再把当前门角加超前量截到规划弧段终点。首次拨门时夹爪世界系方向固定，不随门板旋转。

## GPU与数据边界

策略、物理、候选、逆映射、轨迹及FSM核心数学使用GPU批量计算，不新增逐环境CPU数值求解。USD装配、静态元数据、接触审计、数据打包、Parquet/JSON落盘和视频编码包含必要CPU工作，因此不称整个数采耗时为纯GPU。

本次携臂修复不增加在线碰撞搜索；离线代理单次门框查询6444对以内。初始化/规划的既有复杂度与携臂无关。相同观测的控制数学保持一致，但GPU PhysX跨批量物理轨迹不保证逐bit一致。official恢复已记录状态，不依赖重跑物理一致。

完整开门回归476通过，动力学/锁舌/资产补充62通过、1跳过；迁回后73项针对性回归再次通过。主工作区454个运行文件与已验证临时区字节一致，证据在`artifacts/door_baseline_validation/runtime_migration_equivalence.json`。

## 证据

| 模式 | 当前canonical目录 |
| --- | --- |
| 推门 | `datasets/auto_open/canonical/auto_open_v1_seeds1700-2699_canonical_no_camera_20261103T030040` |
| 拉门 | `datasets/auto_open/canonical/auto_open_v1_seeds1700-2699_canonical_no_camera_20261103T033827` |

各目录含`acceptance_report.json`、`manifest.json`、`workflow_source.json`及全部1000份metadata/shard验收。汇总为`artifacts/door_baseline_validation/final_canonical_summary.json`，推门整批等价核对为`push_canonical_equivalence.json`，拉门增益/退步集合为`pull_canonical_delta.json`。旧入口指纹含绝对源码路径，跨工作区比较另用文件内容校验，不改写原实验来源。

## 抓取与远程采集

外部/背筐抓取复用同一入口、StageSpec编排、控制接口和数据链路，分别选择`AUTO_OPEN_TASK=pick_external|pick_basket`。阶段、场景随机化、成功率/耗时总表和失败分布见[抓取主线说明](pick_gpu_batch_status_zh.md)。最近有效抓取基线为212/400（53.0%）、1085秒，以及313/400（78.25%）、1197秒；共享动力学与资产变更后尚未复测，不冒充本轮新结果。

配套远程采集已切换：wholebody3推门从110000起、wholebody4拉门从210000起；每批两卡各100seed，official模式连续执行新seed。冻结源码提交`b62b0f7`，目录为`.official_service/b62b0f7/source`，附声明且校验哈希的True滤波覆盖，不跟随本地未提交改动。旧服务已停止，旧数据保留。

首批远程canonical：推门175/200（87.5%，1048秒）、拉门161/200（80.5%，1505秒），均进入双鱼眼回放且无监控告警。此时整批official发布数仍为0，图像补录完成和发布由持续服务继续执行；这些首批结果不替代上面的1000seed基线。

服务状态位于`.official_service/b62b0f7/{push,pull}_continuous/status.json`，逐批记录canonical请求数、严格通过数、实际发布数和首因分类，二者不混算；监控每60秒刷新。STOP文件阻止继续分配新批次；立即停止当前批次还需对对应`service.pid`中的容器PID发送TERM。具体部署与首批验收进度维护在[台账](door_baseline_validation_progress_zh.md)。
