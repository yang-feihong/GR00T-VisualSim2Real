# 抓取流程最终验收

## 当前结论

外部和背筐各400条完整canonical已完成，严格成功率分别为53.0%和78.25%，均达到本轮下限。最高成功率、且背筐耗时更短的版本已迁回主工作区。架构/GPU/废代码审核及失败案例证据已完成。推拉门新增受控审查确认：保留授权改动的基线与当前重构版8例25490帧完全一致；原提交到授权版本的退步另行归因于trunk滤波，不能误报为原提交本来如此。剩余抓取能力问题只分析，不继续改动。

| 任务 | 并行方式 | 严格通过/请求数 | 入口墙钟耗时 | 入口至首步 |
| --- | --- | --- | --- | --- |
| 外部抓取 | 4卡，每卡100环境 | 212/400，53.0% | 1085秒，18分05秒 | 405–446秒 |
| 背筐抓取 | 4卡，每卡100环境 | 313/400，78.25% | 1197秒，19分57秒 | 547–764秒 |

分母包括初始化拒绝；均无相机、完整5000步预算，不是串行单环境队列，不包含后续补图。背筐不是每卡都小于10分钟启动。相同313条通过的上一版耗时1509秒，本版减少312秒（20.7%），失败分类逐项不变。

权威统计：

- [外部400条结果](../datasets/pick/.cache/init_speed_gaze_tmp/datasets/pick/canonical/auto_open_pick_external_v1_seeds0-399_canonical_no_camera_20261101T153740/control_outcomes.json)
- [背筐400条结果](../datasets/pick/.cache/init_speed_gaze_tmp/datasets/pick/canonical/auto_open_pick_basket_v1_seeds0-399_canonical_no_camera_20261101T203743/control_outcomes.json)
- [持续维护台账](vlnverse_pick_workflow_zh.md)

## 架构与GPU审核

1. 推拉门比较基线保留此前单独授权的trunk低通滤波和大seed坐标取模，仅要求本次框架重构不改变行为。开门FSM命令、验收、目标规划的原有函数与b56eae4做AST逐函数比较；共享生命周期做CPU/GPU逐值对照，7项通过。抓取重采样需要的策略历史清零已限定为PickEnvironment启用，门环境保持原重置行为。
2. 四种任务共用`run_auto_open.sh`、`batch_workflow.py`、`BatchedTaskLoop.run_stages`、`StageSpec`、`TensorStageLifecycle`、`StagePlanningEvents`、`StageExecutionMixin`和命令缓冲。阶段只决定任务目标、候选及验收。采集共用58维状态/25维动作协议、Parquet/视频写入工具、接触审计及debug/canonical/official的分发和发布；抓取保留场景刚体恢复和抓取语义/验收适配，不另起一套物理步进或串行任务协议。
3. 几何、候选、逆映射、路径、FSM和策略推理在GPU批量执行。没有逐环境核心求解循环。CPU仍用于原生USD资产装配、场景分组与种子元数据、最终诊断/数据序列化、图像处理和编码；不声称零CPU，也不把这些必要边界冒充GPU计算。所有显式主机传输有AST清单及原因审查。
4. 实测加速不是仅将算子搬上GPU：8个真实背筐持物状态，2080个候选完整认证减少为192个，38.066→7.701秒；最终路径、选中候选和失败码逐位一致。仍保留全部260个候选的失败回退，不降低碰撞/逆解/验收条件。完整400条实测也缩短20.7%。
5. 废代码已清理：未调用的7个旧几何/中心修正函数及孤立常量，旧单环境抓取切换/失败重试、旧背筐拨物编排及其入口，以及配置对象中已不用的阶段计时、验收记录、稳定窗口、接触记忆等重复状态。与旧编排绑定的测试已移除或改为检验当前批量FSM；有实际诊断工具调用的离线规划保留。模块可达性、未引用私有函数、未使用导入/局部变量另有检查。清理前后已消费的整套FSM张量、路径缓冲及首次发布命令逐值相同，并留下固定回归签名。

## 推拉门小规模回归

原提交A、A加授权改动B、当前工作区C三组，各运行推门和拉门1700–1703，统一六视角debug入口、相同GPU映射与完整预算。严格结果如下，不是新的1000seed通过率。

| 版本 | 推门严格通过 | 拉门严格通过 |
| --- | ---: | ---: |
| A原提交 | 4/4 | 3/4 |
| B加授权滤波/原点取模 | 0/4 | 3/4 |
| C当前工作区 | 0/4 | 3/4 |

B/C推门11165帧、拉门14325帧全部原始记录逐位相同，包括网络和物理子步，不是只比较通过数。拉门A拒收1702、B/C拒收1703，不能因通过数相同就声称授权改动不改变行为。推门1701在264帧首先出现IK关节目标变化，之后实际状态才变化；原提交峰值门速25.33度/秒、B/C为30.55度/秒，超过原有30度/秒上限。此组seed原点取模不生效，有效控制变量是授权滤波。保留该授权改动，不擅自关闭。

完整失败明细、首次分叉、数据路径及内嵌对照视频见[推拉门受控审核](door_refactor_controlled_audit_zh.md)。抓取历史清零已限定为抓取，门任务恢复原重置语义；没有为通过审核调整门任务目标、轨迹、物理或验收。

## 失败阶段分布

每条按最终失败阶段计一次。

| 外部抓取阶段 | 失败数 | 主要分解 |
| --- | ---: | --- |
| MOVE_TO_GRASP，抓取就位 | 81 | 无安全候选51，到位超时20，异常非足部接触10 |
| LIFT_OBJECT，抬升 | 46 | 失持22，持物路径几何/逆映射拒绝16，到位/抬升超时8 |
| INITIALIZATION，初始化 | 45 | 物理准入拒绝36，GPU无有效初始摆放9 |
| GRASP_OBJECT，夹取 | 7 | 验收超时 |
| APPROACH_OBJECT，靠近 | 6 | 异常接触5，注视路径无解1 |
| WITHDRAW_WITH_OBJECT，持物回收 | 3 | 验收超时 |
| 合计 | 188 | 400−212 |

| 背筐抓取阶段 | 失败数 | 主要分解 |
| --- | ---: | --- |
| INITIALIZATION，初始化 | 47 | 无抓取或安全拨物路径42，无非穿透重摆5 |
| LIFT_OBJECT，抬升 | 31 | 失持28，持物路径几何/逆映射拒绝3 |
| REPOSITION_OBJECT，拨物 | 6 | 无改善方案3，实际拨动超时2，无闭爪侧向进入路径1 |
| CLEAR_FOR_REPOSITION，拨物前避让 | 2 | 张开夹爪密集几何检查拒绝 |
| REOPEN_AFTER_REPOSITION，拨后重新张爪 | 1 | 实际状态无安全夹取候选 |
| 合计 | 87 | 400−313 |

## 主要原因

### 外部抓取就位规划拒绝

51条离线拆分：33条前置夹持/环境几何拒绝，13条没有合格逆映射端点，5条端点逆映射通过但完整路径不通过。33条几何拒绝中28条目标在地面。主因是低处抓取的夹爪/臂姿态与环境、安全路径及规划使用的机身假设耦合，不是单一的“高度够不着”；规划拒绝也不等于wholebody在物理上绝对不可能到达。移动爪保守包络仅解释少数案例，不能作为全部51条的根因。

复跑seed15和31均复现`no_safe_grasp_candidate`。seed31诊断为`no_environment_safe_pocket`。视频显示的是规划拒绝后的停止，不能声称执行了一条碰撞路径。

<video controls width="960" preload="metadata" src="../datasets/pick/debug/auto_open_debug_pick_external_v1_seeds0-167_selected14_episodes2_six_camera_20261101T215520/six_camera_mosaics/seed15_six_camera_mosaic.mp4"></video>

<video controls width="960" preload="metadata" src="../datasets/pick/debug/auto_open_debug_pick_external_v1_seeds0-167_selected14_episodes2_six_camera_20261101T215520/six_camera_mosaics/seed31_six_camera_mosaic.mp4"></video>

### 外部实际到位与抬升不足

不是轨迹还没播完。复跑seed4的MOVE_TO_GRASP路径已完成579帧，末端位置差18.23mm，始终没达到12mm；位置误差单独阻塞551帧，姿态、稳定和场景接触条件都已满足。seed3原本同类超时，本次通过，首次分叉另见下节，不能作为本次改动修好了它。

<video controls width="960" preload="metadata" src="../datasets/pick/debug/auto_open_debug_pick_external_v1_seeds0-167_selected14_episodes2_six_camera_20261101T215520/six_camera_mosaics/seed4_six_camera_mosaic.mp4"></video>

<video controls width="960" preload="metadata" src="../datasets/pick/debug/auto_open_debug_pick_external_v1_seeds0-167_selected14_episodes2_six_camera_20261101T215520/six_camera_mosaics/seed3_six_camera_mosaic.mp4"></video>

seed37复现抬升超时；seed17由原失持变为抬升超时。两者末端位置差约30–36mm，实际净抬升仅约7.1–8.0cm，未达到10cm。这与“没发抬升命令”或“验收误报已经抬够”不同。

<video controls width="960" preload="metadata" src="../datasets/pick/debug/auto_open_debug_pick_external_v1_seeds0-167_selected14_episodes2_six_camera_20261101T215520/six_camera_mosaics/seed37_six_camera_mosaic.mp4"></video>

<video controls width="960" preload="metadata" src="../datasets/pick/debug/auto_open_debug_pick_external_v1_seeds0-167_selected14_episodes2_six_camera_20261101T215520/six_camera_mosaics/seed17_six_camera_mosaic.mp4"></video>

### 外部持物路径拒绝

seed1和8均复现。17个逆映射样本全通过、路径容量合格、终点有正间隙，但初始夹爪几何检查返回穿越/无效间隙哨兵−1，导致整条持物路径被拒绝。−1不是穿透了1米，不能这样解释；该组典型案例明确不是逆映射无解，而是起始夹持几何与障碍检查冲突。尚不能从这一标量确定是支撑面还是家具其他面，需结合碰撞明细，不能笼统称为撞墙。

<video controls width="960" preload="metadata" src="../datasets/pick/debug/auto_open_debug_pick_external_v1_seeds0-167_selected14_episodes2_six_camera_20261101T215520/six_camera_mosaics/seed1_six_camera_mosaic.mp4"></video>

<video controls width="960" preload="metadata" src="../datasets/pick/debug/auto_open_debug_pick_external_v1_seeds0-167_selected14_episodes2_six_camera_20261101T215520/six_camera_mosaics/seed8_six_camera_mosaic.mp4"></video>

### 异常环境接触

外部靠近/就位共15条：10条在机械臂下伸就位，5条在底盘靠近。seed71和167均复现MOVE_TO_GRASP连续3帧非足部场景接触超过20N。raw_contacts已核实这两个典型案例都是`Robot/link06`碰到VLNVerse的`floor`网格，接触对峰值分别43.87N和37.28N，不是碰墙、家具或目标物体。下蹲/低处伸入时，实际腕部碰撞体降到地面，而规划安全和EE中心到位不能保证真实整个腕部同样安全。这是当前低处执行/碰撞包络不够一致的缺陷，不是正常两爪夹持被误判。

<video controls width="960" preload="metadata" src="../datasets/pick/debug/auto_open_debug_pick_external_v1_seeds0-167_selected14_episodes2_six_camera_20261101T215520/six_camera_mosaics/seed71_six_camera_mosaic.mp4"></video>

<video controls width="960" preload="metadata" src="../datasets/pick/debug/auto_open_debug_pick_external_v1_seeds0-167_selected14_episodes2_six_camera_20261101T215520/six_camera_mosaics/seed167_six_camera_mosaic.mp4"></video>

### 初始化拒绝

外部不仅是场景摆放无解：36条在物理预热后被稳定、遮挡、支撑或接触等准入条件拦截；这些子条件可重叠，不应相加成36。背筐主体42条是在实际摆放和机身状态下，没有通过原完整几何/逆解/路径的抓取或安全拨物方案。隔离试验把支撑面6mm改为4mm，在62条状态中恢复0条，因此不能将主体归结为这2mm裕量；原6mm保留。

外部seed19复现初始化拒绝，最终重摆失败；seed7原为初始化拒绝，本次进入夹取后超时。新增初始化六视角记录只观察预热，不改变命令或增加物理步；视频按10帧取1帧、10倍速播放，并保留采样索引，不冒充正式控制帧。

以下外部19、背筐3和7均为真实初始化过程记录。背筐3/7最终全部稳定、支撑及接触检查通过，唯一不合格项是`target_has_grasp_or_safe_reposition_path=false`，并非机器狗失稳。它们没有正式控制帧，不伪造正式控制视频。

<video controls width="960" preload="metadata" src="../datasets/pick/debug/auto_open_debug_pick_external_v1_seeds0-167_selected14_episodes2_six_camera_20261101T215520/logs/rank3-control/seed_19/initialization_six_camera_mosaic.mp4"></video>

<video controls width="960" preload="metadata" src="../datasets/pick/debug/auto_open_debug_pick_basket_v1_seeds0-190_selected6_episodes1_six_camera_20261101T222224/logs/rank1-control/seed_3/initialization_six_camera_mosaic.mp4"></video>

<video controls width="960" preload="metadata" src="../datasets/pick/debug/auto_open_debug_pick_basket_v1_seeds0-190_selected6_episodes1_six_camera_20261101T222224/logs/rank2-control/seed_7/initialization_six_camera_mosaic.mp4"></video>

### 抬升失持

外部22条、背筐28条。背筐28条均已有有效持物规划，不是规划器拒绝。实际夹住后的接触位置、合力线与物体形态/转动稳定性决定能否带离支撑面；不能把规划质量得分等同于真实抗滑/抗转动能力。

背筐最终400条seed0在抬升45帧期间，物体世界高度仅增加约8.1mm，物体中心沿夹爪局部x由67.0移到86.5mm，仍有两侧接触力。因此本批seed0的准确说法是“相对夹持位置明显滑移、未随EE充分抬起后触发失持”，不是复述旧批次的“两侧力归零、滑到105mm”。不能将该单例直接推广为全部28条的同一受力机理。

背筐seed0和15均复现。0最多抬起7.40cm后落回，最终相对夹持位置漂移11.38cm；15最多抬起3.69cm后落回，最终漂移6.43cm，确实没有保住物体。失持判据是物体相对夹爪位置漂移≥15mm连续10帧，不要求两侧接触力先归零；不能把全部触发该代理判据的400条案例都断言为完全脱离。0的实际夹持接触峰值约74.7N/71.9N，不支持简单归因于夹力太小。这些不规则形态的接触几何、力线和支撑反力耦合仍是剩余的实际抓取能力缺口，未擅自调松验收。

<video controls width="960" preload="metadata" src="../datasets/pick/debug/auto_open_debug_pick_basket_v1_seeds0-190_selected6_episodes1_six_camera_20261101T222224/six_camera_mosaics/seed0_six_camera_mosaic.mp4"></video>

<video controls width="960" preload="metadata" src="../datasets/pick/debug/auto_open_debug_pick_basket_v1_seeds0-190_selected6_episodes1_six_camera_20261101T222224/six_camera_mosaics/seed15_six_camera_mosaic.mp4"></video>

外部原失持seed42本次通过，作为未复现对照保留，不把这条视频说成失持；seed17本次则抬升不足，见上文视频。

<video controls width="960" preload="metadata" src="../datasets/pick/debug/auto_open_debug_pick_external_v1_seeds0-167_selected14_episodes2_six_camera_20261101T215520/six_camera_mosaics/seed42_six_camera_mosaic.mp4"></video>

### 背筐拨物没有恢复抓取

该类共6条，追加seed190和42。190复现：第一次拨动后仍无安全夹取路径，第二次剩余侧向空间约44.6mm，但没有任何合格位移能改善抓取/空间，停止于`no_shift_improves_grasp_access_or_space`。有空隙不等于完整张爪、下伸和闭爪路径可通过。42本次通过，原批初始化最终采用attempt2，本次采用attempt6，物体位置和随后路径已不同，不能将成功宣称为针对拨物的修复。

<video controls width="960" preload="metadata" src="../datasets/pick/debug/auto_open_debug_pick_basket_v1_seeds0-190_selected6_episodes1_six_camera_20261101T222224/six_camera_mosaics/seed190_six_camera_mosaic.mp4"></video>

<video controls width="960" preload="metadata" src="../datasets/pick/debug/auto_open_debug_pick_basket_v1_seeds0-190_selected6_episodes1_six_camera_20261101T222224/six_camera_mosaics/seed42_six_camera_mosaic.mp4"></video>

## 未复现案例的首次分叉

外部复跑14条中：seed3、42从失败变为通过，seed7、17、49换了失败类别；其余结果类别保留。原400条统计不因debug结果变化而重写。

补充两条未复现的完整控制视频：seed7从初始化拒绝变为夹取超时，seed49从抬升超时变为抓取就位无安全候选。其余三条未复现视频已嵌入对应类别。

<video controls width="960" preload="metadata" src="../datasets/pick/debug/auto_open_debug_pick_external_v1_seeds0-167_selected14_episodes2_six_camera_20261101T215520/six_camera_mosaics/seed7_six_camera_mosaic.mp4"></video>

<video controls width="960" preload="metadata" src="../datasets/pick/debug/auto_open_debug_pick_external_v1_seeds0-167_selected14_episodes2_six_camera_20261101T215520/six_camera_mosaics/seed49_six_camera_mosaic.mp4"></video>

逐帧比较确认seed3、17、42描述符一致，但第0帧前的根状态/关节/物体状态已不同；seed49最终描述符不同，原运行attempt0接受，本次到attempt2才接受。未仅凭第0帧不同就宣布为PhysX差异，而是追加以下受控对照：

| 对照 | 初始运动学/首次策略与执行输入 | 首次运动学分叉 | 策略输出首次分叉 |
| --- | --- | --- | --- |
| 外部，同描述符14环境与6环境，含3/7/17/42/49 | 一致 | 第1控制步，第1物理子步之后；关节速度约10⁻⁶ rad/s差异 | 第2控制步 |
| 背筐，原始rank2全部100个初始描述符与单独seed42 | 一致 | 第1控制步，子步索引3；关节速度差约0.00596 rad/s | 第2控制步 |

子步索引从0起。记录同时覆盖首次策略观测/输出、关节位置目标、力矩、每个物理子步的关节位置/速度。最初接触传感器缓存部分存在差异，未声称所有Isaac内部状态完全相同；但首个网络输入/输出和首个物理子步前的目标/力矩均一致。运动学分叉先出现在物理推进内部，随后才反馈到控制器，不是相同输入先算出不同控制输出。

背筐42初次用最终布局做6环境/2环境对照，前300步逐位一致，**没有**将其算作已解释；随后重新取原400条的原始100环境初始化描述符，才取得上表的物理子步先分叉证据。其物体接触/稳定演化差异影响何时准入、是否重摆；原批最终采用attempt2，本次attempt6，目标位置相差约11.6cm，后续规划不是同一个输入问题。

历史400条没有保留初始化物理子步，不能倒推出那一次每个浮点量的首次变化；补充实验复现了相关初始条件下的分叉来源，未发现自身控制器先分叉。按用户允许区分控制一致性与PhysX轨迹差异的验收口径保留，不为强行复现失败修改控制或阈值。

证据：[外部初始子步对照](../datasets/pick/.cache/final_divergence_physics_subset6/initialization_comparison.json)、[背筐原始100环境对照](../datasets/pick/.cache/final_basket_original_initial_single42/initialization_comparison.json)。这两组都是诊断探针，不冒充新的canonical成功率。

## 交付范围

本轮新跑外部14条、背筐6条典型失败seed，均走正式debug入口；17条有正式控制六视角视频，另外3条初始化拒绝有独立预热六视角视频。报告已逐类内嵌，不把未复现视频标作原失败。推拉门A/B/C控制变量审核及8例B/C完整原始记录比较已完成，另有内嵌对照视频。远程official服务未改动、未停止。
