# 推拉门受控回归审核

## 结论

本次抓取重构没有在已验证的推拉门样本中增加行为变化：保留授权改动的B组与当前工作区C组，推拉门共8例、25490帧，全部原始记录逐位一致，没有缺失字段。比较包含阶段、底盘与机械臂命令、实际状态、网络输入/输出及四个物理子步的关节状态、力矩和位置目标，不只是比较最终通过数。

相对未引入滤波的原提交，推门确实退步；不能用“基线本来就差”解释。变化在只加入此前授权改动的B组已经完整复现。1700–1703的环境原点取模不生效，因此该对照中的有效授权控制变量是trunk滤波。滤波核心与lab30提交94f1731一致，没有擅自关闭或修改它。

追加反向对照D也已取得完整控制及验收结果：当前代码仅关闭滤波，推门恢复严格4/4、拉门恢复严格3/4；结束步数及拒收seed/原因全部恢复原提交。D/A推门11262帧全部共同控制、运动状态和阶段记录逐位一致，仅1702的原生接触传感器坐标有浮点末位差；拉门14094帧全部字段逐位一致，包含网络及物理子步。主工作区滤波仍开启。这说明本组样本的行为变化由滤波引入，不是抓取重构的额外差异；这不是1000seed的通过率估计。

| 版本 | 推门流程完成 | 推门严格通过 | 拉门流程完成 | 拉门严格通过 |
| --- | ---: | ---: | ---: | ---: |
| A：提交b56eae4 | 4/4 | 4/4 | 4/4 | 3/4 |
| B：A加授权滤波及原点取模 | 2/4 | 0/4 | 4/4 | 3/4 |
| C：当前工作区 | 2/4 | 0/4 | 4/4 | 3/4 |
| D：当前工作区仅关闭滤波 | 4/4 | 4/4 | 4/4 | 3/4 |

每种模式都使用1700–1703，单seed单环境，依次固定在GPU0–3，统一debug入口、六视角、相同目标/物理/策略/验收和完整预算。不是新的1000seed成功率估计。此前“推门1700/1701、拉门1702/1703”的混合四例，在原提交中实际严格3/4，而不是仅1/4。

## 首个分叉

上层仍通过标准接口发布height-invariant EE指令，没有改为发布关节角。下文的世界系EE目标是低层坐标转换后的原始目标，“关节目标”指低层内部IK的输出。

以推门1701为明确因果样本：前264帧实际状态及控制记录一致。第264帧（5.28秒）首次切换EE控制时，原始世界系EE位置目标仍相同，但低层内部IK关节目标最大分量先相差0.0179974rad，FK末端位置最大分量差4.0598mm。实际状态在该步物理推进后才变化；策略输出到下一帧才分叉。这不是先发生底盘偏移、再让机械臂够不到。

滤波没有修改上层目标，而是在低层逆映射转换处将实际机身参考替换为滞后的参考。等效送入IK的世界系目标为`T_actual * inverse(T_filtered) * T_goal`，所以同一个上层EE目标也会产生不同关节目标。两个0.27秒极点同时影响高度、roll和pitch，不仅影响显示或日志。随后接触和门运动轨迹随之改变。

1701原提交峰值门速25.3327度/秒，B/C均为30.5539度/秒，并在相同第2086步因连续三帧超过原有30度/秒上限而失败。B/C该例2086帧的原始记录完全一致。

1702和1703同样先在首次EE控制帧347/388出现关节目标差异，分别0.0106465/0.0066705rad，实际状态和策略输出随后变化。拉门A/B记录亦显示相同的早期分叉顺序。1700在首次正式记录前已有微小初始化差异，不用它单独证明首因，也不假称已记录当时未保存的初始化物理子步。

## 失败明细

| 模式/seed | A原提交 | B与C，相同记录和拒收项 |
| --- | --- | --- |
| 推门1700 | 严格通过 | 完成穿门；第1156帧起阶段外link06/把手接触14条 |
| 推门1701 | 严格通过 | PUSH_PANEL_OPEN超速失败，2086步；其余终态不合格是提前失败的后果 |
| 推门1702 | 严格通过 | ADVANCE_TO_NEXT_PUSH_STATION超时，3272步；接触点相对目标参考中心的水平距离最小0.7053m，仍大于0.45m条件，且机身未稳定 |
| 推门1703 | 严格通过 | 完成穿门；把手操作数据阶段68帧底盘纠偏命令触发禁运动判据 |
| 拉门1700/1701 | 严格通过 | 严格通过 |
| 拉门1702 | 完成穿门；1141帧阶段外link06/把手接触1条 | 严格通过 |
| 拉门1703 | 严格通过 | 完成穿门；阶段外把手接触50条、目标短时往返、66帧底盘纠偏命令、夹爪/把手穿透峰值5.274mm超过3.5mm上限 |

拉门通过数相同并不代表行为不变：拒收从1702变成1703。上述结论没有将授权改动造成的问题掩盖成原始基线失败，也没有用放宽验收来恢复数字。

## 滤波具体改变了什么

滤波作用于低层逆映射使用的实测机身高度、roll、pitch，两个时间常数均为0.27秒的串联极点；所有阶段生效，不仅是行走。xy和yaw未作该低通，上层EE命令本身也未作低通。但是滞后的机身参考会改变同一EE命令对应的机械臂局部目标，因此不能理解为只改变观测显示而不改变控制。0.7m作用距离上，1度倾角差就对应约12.2mm的几何位置差。

机身姿态恒定且滤波收敛后，该转换差消失，并非永久固定偏置。风险出现在走动、下蹲/起身和接触回弹期间：规划使用实际或静态机身几何，没有预计算该动态滞后；执行却使用滞后的参考，可能改变接触位置、方向及施力时序。这个机制不等于对上层EE快速命令做平滑，也不能保证只抑制无害抖动。

以下误差是内部关节目标FK位置与原始世界系EE目标之间的距离，包含滤波、IK及反馈约束的综合影响，不能全部称为纯滤波位移。原始量和分段统计保存在`datasets/auto_open/.cache/filter_behavior_metrics.json`。

| 样本 | 原提交与滤波版的实测差别 | 对失败机理的解释与边界 |
| --- | --- | --- |
| 推门1700 | 撤手阶段内部目标误差峰值2.12→16.58mm；滤波版14帧link06/把手接触，峰值25.38N | 滞后转换可能消耗撤手间隙；已确认碰把手，不是猜测碰墙，尚未逐接触证明全部作用链 |
| 推门1701 | 第二轮推门内部目标误差中位数2.71→13.73mm、峰值2.90→26.66mm；实际门速峰值25.33→30.55度/秒 | 改变施力轨迹和接触状态，触发连续3帧超过30度/秒的原有判据。两组此轮起始门角也不同，不能当作相同门角的局部力学实验 |
| 推门1702 | 第二轮离手门速21.33→16.44度/秒；随后收臂末门角88.91→80.11度 | 原提交门仍在转，随后超过89度进入穿门；滤波版停在80.11度，多进入一轮靠近并超时。不是同一轮目标被突然收紧。滤波改变了先前动作和分支入口状态；较小离手动量是合理解释，收臂期间可能仍有接触，不能只用自由转动阻尼解释 |
| 推门1703 | 开门把手阶段内部目标误差峰值1.06→27.08mm；机身高度波动范围1.62→31.41mm，阶段起点起算的xy偏移峰值27.61→73.60mm | 滞后参考、接触反力和机身姿态可能形成耦合，站位保持接口发出纠偏，触发数据禁运动判据 |
| 拉门1703 | 松爪内部目标误差峰值2.67→28.88mm；夹爪/把手穿透峰值1.321→5.274mm；撤手误差峰值1.92→29.54mm | 第1145帧q4内部目标跳变0.07351rad，1146帧返回，与首次阶段外接触同帧。目标在物理推进前计算，不能反过来断言接触冲击导致此帧目标跳变 |

`base_moved_during_handle_press`名称容易误导。实际数据判据是把手操作数据阶段（MOVE_TO_PREGRASP至STOW_ARM_AND_CLOSE_GRIPPER）的`max(abs(base_commands)) > 0.03`，不是实际位移阈值，也不限于下压阶段。推门1703的68帧全在OPEN_DOOR_WITH_HANDLE，首帧890，命令为[-0.14225, 0.04758, 0.30329]；拉门1703的66帧分为RESET_HANDLE 10、RELEASE_HANDLE 47、WITHDRAW_FROM_HANDLE 9，首帧1088。原提交两例均0帧。拉门穿透第一次超限是4.258mm，峰值才是5.274mm，共9帧，均在RELEASE_HANDLE。

## 当前代码关闭滤波的反向对照

按追加要求运行D组：复制当前全部运行源码，仅将`TRUNK_CONTROL_FILTER_ENABLED`设为False，不改目标、轨迹、物理或验收。主工作区仍为True。离线3项检查确认运行源码只有该开关不同、关闭时直接返回原始目标张量且不更新历史、模型和资产一致。

D推门完成严格4/4，结束步数2893/2751/2793/2825与A全部相同。11262帧全部共同控制/状态/阶段字段逐位一致；1702只有传感器接触点坐标记录存在浮点末位差，首次在post记录1107帧、pre记录1108帧，差1.19209e-7m，未传播至任何控制或运动状态。坐标直接取自Isaac接触传感器`contact_pos_w`，没有自己的接触点算术转换。A推门没有网络/物理子步扩展字段，不把这些缺失项算作已比较。

D拉门完成严格3/4，结束步数3482/3340/3765/3507与A相同。1700/1701/1703严格通过；1702恢复A的唯一拒收项：第1141帧link06/把手阶段外接触1条。四例14094帧全部原始字段逐位一致，无缺失字段，包含网络观测/动作和四个物理子步的状态/位置目标/力矩。因此没有需要继续归因的控制或物理轨迹分叉。

证据分别为D推门和拉门批次的`door_comparison.json`，完整相对路径列于下表。推拉门共25356帧；没有放宽任何标准来得到这些结果。结合B/C对照，本轮8例中滤波是充分的行为变化因素，关闭它也恢复原提交；不能据此保证所有seed的门任务均不再受其他因素影响。

## 修复边界

审查发现，抓取重采样需要的策略历史清零曾扩散到门任务。现已将基础环境的`reset_policy_history_on_reset`设为False，仅PickEnvironment启用，恢复门任务原重置语义。这是应纠正的越界修改，但不能把发现它等同于证明它解释了先前全部失败；本轮C使用修正后的代码，且与B完整逐位一致。

没有进一步修改门任务选点、轨迹、控制参数、动力学或验收。此前授权的滤波和大seed原点取模仍保留。本次补充仅涉及离线报告、测试、台账及隔离对照区的诊断记录。

## 证据与耗时

批次根目录位于仓库`datasets/auto_open`。A/B根目录另加`.cache/door_audit_A/datasets/auto_open`、`.cache/door_audit_B/datasets/auto_open`前缀。

| 组/模式 | debug批次名 | 采集墙钟秒 |
| --- | --- | ---: |
| A推门 | auto_open_debug_push_v1_seeds1700-1703_episodes4_six_camera_20261101T235000 | 1034 |
| A拉门 | auto_open_debug_pull_v1_seeds1700-1703_episodes3_six_camera_20261102T004319 | 2484 |
| B推门 | auto_open_debug_push_v1_seeds1700-1703_episodes0_six_camera_20261102T004318 | 2258 |
| B拉门 | auto_open_debug_pull_v1_seeds1700-1703_episodes3_six_camera_20261102T004321 | 2475 |
| C推门 | auto_open_debug_push_v1_seeds1700-1703_episodes0_six_camera_20261102T001034 | 2366 |
| C拉门 | auto_open_debug_pull_v1_seeds1700-1703_episodes3_six_camera_20261102T002424 | 3101 |
| D推门 | auto_open_debug_push_v1_seeds1700-1703_episodes4_six_camera_20261102T080827 | 1279 |
| D拉门 | auto_open_debug_pull_v1_seeds1700-1703_episodes3_six_camera_20261102T080828 | 1574 |

这些debug批次并发占用GPU并渲染六视角，墙钟秒不是独占canonical性能比较，不能据此推断重构增加了规划耗时。机器人主USD三组SHA256均为`d300b3242e93e97906a330ee4e57b8e68cc59087d4f3e933b9894a9f37913476`，checkpoint共享同一文件。

C两个批次的`door_comparison.json`保存B/C全字段比较，推门11165帧、拉门14325帧，所有`first_exact_difference`为空。A/C推门比较保存在`datasets/auto_open/.cache/door_A_C_push_comparison.json`；A/B拉门比较在B拉门批次`door_comparison.json`。

D批次根目录前缀为`datasets/auto_open/.cache/door_audit_D_filter_off/datasets/auto_open/debug/`：

| 模式 | D批次名 | 比较结果 |
| --- | --- | --- |
| 推门 | auto_open_debug_push_v1_seeds1700-1703_episodes4_six_camera_20261102T080827 | 11262帧共同控制/状态/阶段逐位一致，仅原生传感器坐标末位差 |
| 拉门 | auto_open_debug_pull_v1_seeds1700-1703_episodes3_six_camera_20261102T080828 | 14094帧全部字段逐位一致，无缺失 |

隔离区最初缺少忽略的材质资源，首次启动无有效episode；补齐共享资源后重跑。扩展诊断也曾遗漏旧writer的二维字段序列化，造成B1701完成控制后落盘失败；该批及随后中止的无效批次不计入结果。修复仅限隔离区诊断写盘，直接抽取两个实际writer的离线往返测试通过后才重启。最终表中六批均有完整四条验收报告，不使用失败批次冒充完整交付。A最初推门没有扩展网络/子步字段，只在其实际保存字段上比较；B/C八例均包含完整扩展字段。

## 视频

以下为当前工作区的五种拒收现象，均来自完整统一debug入口。A原提交参照视频同时给出，所有批次的其余视频均位于各自`six_camera_mosaics`目录。

推门1700：完成流程后因阶段外把手接触拒收。

<video controls preload="none" src="../datasets/auto_open/debug/auto_open_debug_push_v1_seeds1700-1703_episodes0_six_camera_20261102T001034/six_camera_mosaics/seed1700_six_camera_mosaic.mp4"></video>

推门1701：原提交通过，滤波版本超速失败。

<video controls preload="none" src="../datasets/auto_open/.cache/door_audit_A/datasets/auto_open/debug/auto_open_debug_push_v1_seeds1700-1703_episodes4_six_camera_20261101T235000/six_camera_mosaics/seed1701_six_camera_mosaic.mp4"></video>
<video controls preload="none" src="../datasets/auto_open/debug/auto_open_debug_push_v1_seeds1700-1703_episodes0_six_camera_20261102T001034/six_camera_mosaics/seed1701_six_camera_mosaic.mp4"></video>

推门1702：下一推门站位移动超时。

<video controls preload="none" src="../datasets/auto_open/debug/auto_open_debug_push_v1_seeds1700-1703_episodes0_six_camera_20261102T001034/six_camera_mosaics/seed1702_six_camera_mosaic.mp4"></video>

推门1703：完成流程，但把手操作数据阶段发布了不允许的底盘纠偏命令。

<video controls preload="none" src="../datasets/auto_open/debug/auto_open_debug_push_v1_seeds1700-1703_episodes0_six_camera_20261102T001034/six_camera_mosaics/seed1703_six_camera_mosaic.mp4"></video>

拉门1703：原提交严格通过，滤波版本完成流程但严格拒收。

<video controls preload="none" src="../datasets/auto_open/.cache/door_audit_A/datasets/auto_open/debug/auto_open_debug_pull_v1_seeds1700-1703_episodes3_six_camera_20261102T004319/six_camera_mosaics/seed1703_six_camera_mosaic.mp4"></video>
<video controls preload="none" src="../datasets/auto_open/debug/auto_open_debug_pull_v1_seeds1700-1703_episodes3_six_camera_20261102T002424/six_camera_mosaics/seed1703_six_camera_mosaic.mp4"></video>

### 当前代码仅关闭滤波

以下D组与上方A/C形成反向对照。四个推门均严格通过；拉门仅1702因原提交同帧同接触拒收，不把它隐去。

D两个debug入口均正常退出，八条拼接视频齐全。本报告与抓取最终报告的35条内嵌视频已离线逐条检查存在、可解码且非空白（1项检查通过），没有追加闭环测试。

推门1700：

<video controls preload="none" src="../datasets/auto_open/.cache/door_audit_D_filter_off/datasets/auto_open/debug/auto_open_debug_push_v1_seeds1700-1703_episodes4_six_camera_20261102T080827/six_camera_mosaics/seed1700_six_camera_mosaic.mp4"></video>

推门1701：

<video controls preload="none" src="../datasets/auto_open/.cache/door_audit_D_filter_off/datasets/auto_open/debug/auto_open_debug_push_v1_seeds1700-1703_episodes4_six_camera_20261102T080827/six_camera_mosaics/seed1701_six_camera_mosaic.mp4"></video>

推门1702：

<video controls preload="none" src="../datasets/auto_open/.cache/door_audit_D_filter_off/datasets/auto_open/debug/auto_open_debug_push_v1_seeds1700-1703_episodes4_six_camera_20261102T080827/six_camera_mosaics/seed1702_six_camera_mosaic.mp4"></video>

推门1703：

<video controls preload="none" src="../datasets/auto_open/.cache/door_audit_D_filter_off/datasets/auto_open/debug/auto_open_debug_push_v1_seeds1700-1703_episodes4_six_camera_20261102T080827/six_camera_mosaics/seed1703_six_camera_mosaic.mp4"></video>

拉门1700：

<video controls preload="none" src="../datasets/auto_open/.cache/door_audit_D_filter_off/datasets/auto_open/debug/auto_open_debug_pull_v1_seeds1700-1703_episodes3_six_camera_20261102T080828/six_camera_mosaics/seed1700_six_camera_mosaic.mp4"></video>

拉门1701：

<video controls preload="none" src="../datasets/auto_open/.cache/door_audit_D_filter_off/datasets/auto_open/debug/auto_open_debug_pull_v1_seeds1700-1703_episodes3_six_camera_20261102T080828/six_camera_mosaics/seed1701_six_camera_mosaic.mp4"></video>

拉门1702：

<video controls preload="none" src="../datasets/auto_open/.cache/door_audit_D_filter_off/datasets/auto_open/debug/auto_open_debug_pull_v1_seeds1700-1703_episodes3_six_camera_20261102T080828/six_camera_mosaics/seed1702_six_camera_mosaic.mp4"></video>

拉门1703：

<video controls preload="none" src="../datasets/auto_open/.cache/door_audit_D_filter_off/datasets/auto_open/debug/auto_open_debug_pull_v1_seeds1700-1703_episodes3_six_camera_20261102T080828/six_camera_mosaics/seed1703_six_camera_mosaic.mp4"></video>
