# 抓取当前版本说明

本文是外部物体与背筐抓取的主线说明：维护共用入口、场景随机化、阶段定义、最近有效能力和主要缺口，不堆叠历史修改记录。

## 当前能力

成功率按请求的全部seed计数，包含初始化拒绝，使用严格数据验收，不是仅统计已进入控制的样本。以下是最近有效完整canonical基线，不含相机及后续补图。

| 任务 | 规模 | 严格通过率 | 总耗时 | 入口至首步 |
| --- | --- | --- | --- | --- |
| 外部抓取 | 4卡×100环境，seed0–399 | 212/400，53.0% | 1085秒，18分05秒 | 405–446秒 |
| 背筐抓取 | 4卡×100环境，seed0–399 | 313/400，78.25% | 1197秒，19分57秒 | 547–764秒 |

后续资产扩展、夹取规划裕量统一，以及本轮共享机械臂动力学、摩擦及控制修改之后，抓取尚未重新运行400seed；上表是最近有效能力基线，不能宣称当前工作区已复测达到同一成功率。本轮优先验收推拉门，不擅自启动新的抓取大批次。

权威证据、失败案例视频及首次分叉分析见[抓取验收报告](pick_final_acceptance_zh.md)，持续实验记录见[抓取台账](vlnverse_pick_workflow_zh.md)。本轮开门发布门槛见[推拉门验收台账](door_baseline_validation_progress_zh.md)。

## 共用入口

四类任务共用`gr00t/rl/scripts/run_auto_open.sh`，抓取仅切换任务参数，不另设入口或采集协议。

| 参数 | 值与含义 |
| --- | --- |
| AUTO_OPEN_TASK | `pick_external`外部，`pick_basket`背筐 |
| AUTO_OPEN_MODE=debug | 六相机记录、严格验收、六视角拼接视频，不发布official |
| AUTO_OPEN_MODE=canonical | 无图像完整物理执行、严格验收，不进入补图链路 |
| AUTO_OPEN_MODE=official | canonical筛选后，状态恢复双鱼眼补图、复验及正式发布 |
| GPU_IDS | 物理GPU列表，每卡单进程、多个独立环境批量执行 |
| SEED_START/SEED_END | 闭区间，每个seed对应独立episode |

```bash
AUTO_OPEN_TASK=pick_external AUTO_OPEN_MODE=debug GPU_IDS=0,1,2,3 \
SEED_START=1700 SEED_END=1703 bash gr00t/rl/scripts/run_auto_open.sh
```

外部400条canonical：改为`AUTO_OPEN_MODE=canonical SEED_START=0 SEED_END=399`。背筐再改为`AUTO_OPEN_TASK=pick_basket`。默认完整预算5000步，不裁短后冒充成功率。输出在`datasets/pick/{debug,canonical,releases,work}`，任务名称进入批次目录，结构与开门一致。

六视角包括机身鱼眼、腕部鱼眼、机身正上方、物体侧面、物体上方和机身后方。正式数据沿用58维状态、25维动作等共享格式；另记录目标物体标签、资产、支撑物/支撑面、摆放与场景描述及干扰物，支持唯一任务语义和状态恢复补图。

## 场景与随机化

- 使用VLNVerse场景，保留原始场景碰撞。场景选择使用`seed // 20`；连续20个episode共享场景资产，不共享机器人、物体摆放、策略状态或物理交互，不能互相看见。
- 外部目标可置于地面或桌、椅、沙发等可达支撑面，靠边摆放；头部相机到目标初始距离1–3m，朝向偏差不超过60°，通过可见性、支撑、碰撞及物理预热准入。机械臂以front preset随机化。
- 背筐任务随机数量、类别、位置，机械臂以back preset随机化，底盘保持站位。拥挤或靠边时允许闭爪拨物后重抓，不靠关闭碰撞通过。
- 干扰物的种类、数量和支撑位置随机，避免目标类别加支撑标签无法区分。资产按实际尺寸、夹爪可夹持维度及几何准入筛选，不能仅凭类别名判定可抓。
- 资产和来源见[物体目录](pick_object_catalog_zh.md)、[Lightwheel](lightwheel_pick_assets_zh.md)、[Extwin](extwin_pick_assets_zh.md)及`gr00t/rl/pick/asset_sources.json`。受限资产仅用于已确认的非商业研究用途。
- 外部/背筐的夹取规划间隙共用`GRASP_CLEARANCE_M`，当前6mm，不分两套标准；目标间距与实际接触验收的区别见[夹取间隙说明](grasp_clearance_zh.md)。

## 外部抓取阶段

阶段名称、transition map和控制模式的权威定义在`gr00t/rl/pick/stages.py`；批量控制与验收在`tensor_fsm.py`，规划在`tensor_planning.py`。

| 阶段 | 行为 | 主要完成条件 |
| --- | --- | --- |
| INITIALIZATION | 场景/物体/底盘/初始臂姿随机化，物理预热与规划准入 | 支撑、可见性、稳定、碰撞及必要路径可行 |
| APPROACH_OBJECT | 底盘沿预计算轨迹靠近；EE沿注视轨迹跟随物体，站位与注视有扰动 | 底盘到位且稳定 |
| MOVE_TO_GRASP | 保持底盘，选择可行夹取方向与路径，让物体进入夹爪凹槽 | EE轨迹完成、实际位置/姿态准确且稳定 |
| GRASP_OBJECT | 保持EE、闭爪 | 双侧有效夹持接触及力、稳定窗口满足 |
| LIFT_OBJECT | 持物抬升 | 净抬升至少10cm，物体高度至少离地30cm，持物相对位姿不丢失，EE到位 |
| WITHDRAW_WITH_OBJECT | 沿持物安全轨迹适度回收，避开原支撑面 | EE到位、物体保持抬升并仍被夹持 |
| VERIFY_RETENTION | 持物保持 | 双侧接触、物体和EE稳定、抬升/高度满足，连续50帧 |
| SUCCESS/FAILURE | 终态记录与数据验收 | 流程成功仍须严格数据验收通过 |

低处目标通过wholebody EE指令触发自然下蹲，不能用正常站姿静态臂IK不可达作为拒绝低处命令的理由。抬升要求同时约束净抬升和离地高度：低桌面也必须最终达到离地30cm，不是按地面/桌面标签二选一。

## 背筐抓取阶段

从`MOVE_TO_GRASP`进入夹取，底盘不执行外部靠近运动；共用抓取、抬升和保持阶段。抬升后直接进入`VERIFY_RETENTION`，额外要求物体底部高于筐沿10cm。需要拨物时增加以下闭环，仍使用共享标准控制接口。

| 阶段 | 行为与完成条件 |
| --- | --- |
| OPEN_FOR_REPOSITION | 保持EE并张爪，待张开稳定 |
| CLEAR_FOR_REPOSITION | 执行避让EE轨迹，到安全位置稳定 |
| CLOSE_FOR_REPOSITION | 保持EE并闭爪，待闭合稳定 |
| REPOSITION_OBJECT | 闭爪沿规划轨迹轻拨，改善目标两侧插入空间 |
| REOPEN_AFTER_REPOSITION | 张爪并重新检查安全夹取方案；返回抓取或必要的下一轮拨物 |

## 架构与GPU边界

复用开门的`StageSpec`、`StageInterface`、`TensorStageLifecycle`、`StageExecutionMixin`、`BatchedTaskLoop`、命令缓冲和稳定窗口。阶段只定义目标/候选/轨迹及任务验收，底盘使用单点或预计算轨迹，机械臂使用EE单点/轨迹等标准接口，不自建逐环境控制循环。

核心候选几何、逆映射、轨迹、FSM及策略推理在GPU批量执行；CPU负责USD装配、场景分组/元数据、最终序列化和视频编码等必要边界，不声称零CPU。相同场景复用几何缓存。初始化采用有界联合候选采样，不将站位、姿态、误差、方向和初值全部笛卡尔展开。完整审核及实测加速证据在验收报告。

## 最近有效失败分布

每条按最终失败阶段统计一次。

| 外部阶段 | 失败数 | 主要原因 |
| --- | ---: | --- |
| MOVE_TO_GRASP | 81 | 无安全候选51，到位超时20，非足部异常接触10 |
| LIFT_OBJECT | 46 | 失持22，持物路径拒绝16，到位/抬升超时8 |
| INITIALIZATION | 45 | 物理准入拒绝36，无有效初始摆放9 |
| GRASP_OBJECT | 7 | 夹取验收超时 |
| APPROACH_OBJECT | 6 | 异常接触5，注视路径无解1 |
| WITHDRAW_WITH_OBJECT | 3 | 持物回收超时 |
| 合计 | 188 | 400−212 |

| 背筐阶段 | 失败数 | 主要原因 |
| --- | ---: | --- |
| INITIALIZATION | 47 | 无抓取/安全拨物路径42，无非穿透重摆5 |
| LIFT_OBJECT | 31 | 失持28，持物路径拒绝3 |
| REPOSITION_OBJECT | 6 | 无改善方案3，拨动超时2，无安全闭爪进入路径1 |
| CLEAR_FOR_REPOSITION | 2 | 张爪密集几何拒绝 |
| REOPEN_AFTER_REPOSITION | 1 | 实际状态无安全夹取候选 |
| 合计 | 87 | 400−313 |

主要能力缺口是低处抓取的环境/wholebody可行性与实际到位、非规则物体抗滑/抗转动持物稳定性、密集背筐摆放下的安全进入与拨物可行性。规划通过不等于物理抓稳；规划拒绝也不等于物理绝对不可达。部分debug未复现原失败的首个分叉已定位在PhysX物理子步，不能冒称控制器修复效果。具体原因、证据和视频以[验收报告](pick_final_acceptance_zh.md)为准。
