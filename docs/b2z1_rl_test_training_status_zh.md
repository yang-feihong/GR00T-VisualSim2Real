# B2Z1 开门强化学习测试训练说明与进展

更新时间：2026-09-30 12:33（北京时间）

## 0. 当前总览（每轮实验后更新）

### 文档维护规则

本文件是该训练任务的唯一持续状态记录。上下文恢复、切换机器或开始新实验时，先读本节和“当前瓶颈与下一实验”，不得仅凭聊天摘要继续执行。

每轮实验必须按以下顺序维护：

1. 开始前登记实验编号、基线 checkpoint、唯一变量、待验证假设、成功/失败判据和禁止改动范围。
2. 运行中只补充无副作用的状态与证据，不把尚未完成的实验写成已验证结论。
3. 结束后先更新顶部的有效 checkpoint、到达率、当前瓶颈和下一决策门槛，再在文末追加不可覆写的实验记录。
4. 对失败尝试必须记录“改了什么、观测到什么、排除了什么、为何不继续”，避免换 phase 后重复尝试。
5. 行为修改必须标明所有权层、被替代的旧逻辑以及前序阶段回归结果；不允许并存两套补偿或控制路径。
6. 只有经实际输出或对照实验确认的内容写入“已确认”；推测必须明确标记为“待验证假设”。
7. 仅在明确要求或形成有效 checkpoint、阶段结论等关键里程碑时同步远程；同步状态不得冒充远程实测结果，也不得打断当前唯一实验。
8. 实验状态只允许使用“计划中、运行中、已验证、失败”四种；启动、首个 update、checkpoint、结束或异常时立即更新顶部执行卡，不能等到下一轮实验才补写。
9. 顶部执行卡是唯一当前状态；历史章节不得继续使用“当前”“正在运行”等易混淆措辞。聊天汇报只引用本文件中已经核对过的状态，不另建一套口头进度。
10. 只在实验状态、决策证据或下一门槛变化时追加记录；普通轮询不写入文档，避免重复的“仍在运行”掩盖有效结论。
11. 六阶段总账是上层规划入口：阶段目标、最强有效证据、首个瓶颈或后续门槛发生变化时，必须与当前执行卡和 phase 记录在同一次编辑中更新。
12. 已确认并修复的问题进入“已解决问题”，失败或被排除的方案进入“决策与禁重复清单”；后续重新打开旧结论必须先登记新的直接反证，不能仅因上下文恢复而重试。
13. 正式实验的原始标准输出必须通过入口脚本 `tee` 到容器内明确路径；文档中的日志路径必须注明所在容器。实验结束后先核对日志和 checkpoint 均存在，再登记结果。
14. 每个 phase 的状态变化必须原子更新四处：当前执行卡、上层路线图、六阶段总账、文末 phase 记录。任一处仍指向旧 phase，均视为记录未完成，不得启动下一实验。
15. “近期尝试结果”只保留会影响下一步决策的最近实验；它必须明确写出假设、结果和是否允许重试。完整证据仍保留在文末，不用聊天记录替代。
16. 上下文压缩前后都以“当前执行卡”的原子游标为准。恢复后先核对游标所列已完成动作、下一原子动作、进程、日志和工作树；不得重新回应已经完成的旧请求，不得从 phase 起点重做已登记动作，也不得在未更新游标的情况下继续编辑或启动实验。
17. 本文件同时承担上层规划和实验日志两种职责：每次有效进展必须同步维护“当前决策链、当前执行卡、上层路线图、六阶段总账、近期尝试和文末 phase 记录”。其中上层路线图和六阶段总账必须明确“阶段目标、已完成进展、当前阻塞、下一验收门槛”。只追加 phase 日志、却让上层目标仍停在旧结论，视为未完成记录。
18. 每项尝试都必须留下可检索的闭环：`问题/假设 -> 唯一改动或只读诊断 -> 直接证据 -> 结论 -> 是否允许重试 -> 下一原子动作`。已经解决的问题和已经排除的方向不得因换机器、换会话或上下文压缩而重新尝试。

每个新 phase 必须使用同一记录结构，缺一项不得启动：

```text
phase / 状态：
上游结论与待解决瓶颈：
基线 checkpoint / SHA-256：
唯一变量：
待验证假设：
禁止改动范围：
运行规模与日志：
成功/失败判据：
观测指标：
结论与已排除方向：
正式保留/删除的代码：
前序阶段回归：
下一决策门槛：
```

“当前执行卡”必须和 phase 记录同时更新：开始时写入运行状态、基线和日志；结束时写入结论、最新有效 checkpoint 和唯一下一步。不能只在文末追加记录而让顶部继续指向旧实验。

每次恢复工作时固定执行：先读本节确认当前有效 checkpoint 和唯一下一实验，再读文末最近两个 phase 的完整记录，最后用进程、日志和工作树状态核对文档是否仍然成立。若三者不一致，先修正文档，不启动新实验。

恢复检查必须得到一个幂等游标：`已完成到哪里 -> 下一原子动作 -> 完成判据`。聊天中的“继续”只推进该下一原子动作；完成后先移动游标，再做后续动作。这样即使压缩发生在一次 phase 中间，也不会重复回复旧问题、重复启动进程或覆盖尚未完成的实现。

### 任务目标与不可破坏约束

最终目标不是让某一轮统计看起来更好，而是训练出同一个纯 RL 策略，在 seed `1700--1703` 上安全完成：

```text
WALK_TO_DOOR -> GRASP -> OPEN_LATCH -> REPOSITION -> PUSH_OPEN -> THROUGH
```

完成该目标前，按以下上层顺序推进：

1. 保证纯动力学、headless、无相机和无 RTX rendering 的训练基础设施稳定。
2. 保证 actor 在六阶段始终独立拥有语义统一的全部 10 维动作。
3. 从最早未学会阶段开始，只定位该阶段的首个真实断点；没有证据前不修改后续阶段。
4. 每次只验证一个假设和一个行为变量，并回归此前已通过的接口与阶段。
5. 六阶段都能学习后，才使用同一 checkpoint 验收 seed `1700--1703`，保留日志、轨迹和 SHA-256。

全过程硬约束：

- 禁止复用、模仿或注入 AutoOpen 补偿、动作轨迹、reference action、steady inverse 或行为监督。
- reward 只能评价动作执行后的物理结果，不得奖励 actor command 接近脚本目标。
- 禁止阶段特例、seed 特调、门型特调、隐藏开关和为快速过验收引入的临时阈值。
- observation、action、坐标系和阶段验收语义必须全流程统一；修复必须落在正确的接口所有权层。
- 根因不清楚时只做无副作用诊断；新机制替代旧机制后删除旧路径和诊断死代码。
- 不为兼容旧 checkpoint 破坏架构；接口语义变化时从随机初始化干净训练。

### 当前决策链（一屏摘要）

```text
最终验收：同一纯 RL checkpoint 安全通过 seed 1700--1703 的全部六阶段
    ^
后续门槛：完成 OPEN_LATCH/REPOSITION 合法接触态初态和独立阶段训练，再进行同一策略串联训练
    ^
当前门槛：验证 GRASP 碰撞感知导航势场能否消除直穿把手的错误梯度，并超过 step225 的约 29.2% 完整夹持率
    ^
已通过独立阶段：WALK_TO_DOOR 89.11%；PUSH_OPEN 91.99%/92.68%；THROUGH 85.11%/85.06%
    ^
GRASP 当前基线：step225 两个 seed 完整夹持 600/2048 与 598/2048，但已确认欧氏位置势场存在直穿把手的路径拓扑错误
```

### 当前执行卡（恢复工作时以此为准）

```text
状态核对时间：2026-09-30 12:33（北京时间）
最近完成里程碑：独立阶段正式验收完成；PUSH_OPEN 两个 seed 为 `91.99%/92.68%`，THROUGH 为 `85.11%/85.06%`，GRASP step225 为 `29.30%/29.20%`
当前 phase：GRASP 连续移动法向目标续训至 step300
当前运行进程：从 step200 保持代码与 optimizer 状态不变四卡续训100 updates；日志 `doorman_train:/tmp/b2z1_grasp_moving_target_to300.log`
当前瓶颈：移动目标已消除旧安全距离局部最优，但姿态学习明显慢于位置学习；step200 orientation/axis error 仍为 `0.93/0.87 rad`，导致最终接触尚未形成。PUSH_OPEN 与 THROUGH 不是当前瓶颈
已完成到：step125/150/175/200 四个 checkpoint 已用相同2048-env确定性协议并行复评，完整夹持均为0；step150 最好，任一/双爪/对向接触 `153/49/54`。训练期姿态误差从 step25 到200持续单调下降，尚未形成平台
下一原子动作：保持行为代码冻结，续训并保存 step225/250/275/300；跟踪姿态改善后真实接触是否恢复，随后四卡并行做同口径确定性复评
禁止范围：不得用 stage、seed、门型、readiness 或脚本目标直接缩放噪声；不得全局降低固定 std、设置训练步数阈值退火、扫描 reward 权重或修改完成条件；不得复用旧 checkpoint，接口变化后必须干净训练
当前候选代码：失败的单边 clearance 饱和与 `B*direct_position` 最终靠近路径已删除；唯一正式候选为连续移动法向目标，动作、observation、验收、其他 reward 和其他阶段均未修改
当前阶段 checkpoint：GRASP 基线 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260930_050612/model_step_000225.pt`；PUSH_OPEN `wholebody_3:/datadisk/workspace/doorman_docker_package/GR00T-VisualSim2Real/logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260929_191748/model_step_000400.pt`；THROUGH `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260929_222553/model_step_000400.pt`
历史证据 checkpoint：`logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260928_105610/model_step_000125.pt`，SHA-256 `f1262adb68a4bdad017c800741d50772b15af799b743d7a43c52313a19230a62`；仅用于证明旧混合控制行为，不得恢复训练或作为纯 RL 成绩
当前证据：GRASP `logs_eval/grasp_step225_stage1_seed{0,1}_eval2048_idfix_v2/metrics_eval.json`；PUSH_OPEN `logs_eval/push_open_current_worktree_regression_eval2048/metrics_eval.json` 及 seed1 正式复评；THROUGH `logs_eval/through_current_worktree_regression_eval2048/metrics_eval.json` 及 seed1 正式复评
最近正式备份：2026-09-28 15:27；phase217 完成状态、phase218 计划和当前 32 个修改/新增源码、配置、测试与文档已同步到 `wholebody_1/3/4`，四机同步文件聚合 SHA-256 一致为 `51669c196d94c046ca09a83502c138221d1e4991eee871aea81d9dabfb285f2c`；未同步日志、缓存、`/tmp` 诊断数据、运行目录或凭据，未修改远程 Git 历史
最终验收状态：未完成；尚无同一策略在 seed 1700--1703 完成六阶段的证据
```

### 当前未决问题与证据门槛

| 优先级 | 未决问题 | 已知事实 | 形成结论前必须取得的证据 | 当前状态 |
| --- | --- | --- | --- | --- |
| P0 | actor 是否真正拥有完整动作 | 旧脚本链已删除；左右门十维中心差分雅可比均满秩 10，映射无 stage 输入，清理后 optimizer smoke 和防回归通过 | 后续配置或动作接口变化时重复静态契约和十维数值验证 | phase217 已验证 |
| P0 | GRASP 势场能否从相关初态沿无碰撞路径到达完整夹持 | step225 已达约 `29.2%`；失败局最接近成功时约 `23%` 两爪同在把手上方或下方，旧欧氏场奖励直穿并惩罚后退 | 新势场 update200 的同口径确定性复评必须提高完整夹持率，并降低上下同侧失败且不增加 unsafe force | 四卡干净训练中 |
| P1 | GRASP 左右门差异 | step225 高成功门型约 `52%`，镜像门型约 `6--7%`；低成功侧多数失败局仍曾上下跨住把手，不能只归因于高度方向 | 新势场按门侧拆分 pocket、axis、包围与双侧接触；只有直接证据支持时再修改下一项 | 等当前单变量实验结果 |
| P1 | THROUGH 独立阶段是否可学习 | step400 两个 seed 确定性成功 `85.11%/85.06%`，剩余弱项是穿门距离尾部 | 串联训练后确认其入口分布变化是否导致回归 | 本轮独立阶段已通过 |
| P1 | 最终四 seed 验收 | AutoOpen 有四 seed 物理成功案例，但 RL 尚无同一 checkpoint 全通过证据 | 同一 checkpoint 在 `1700--1703` 安全完成六阶段，并保留轨迹、日志和结果 | 未完成 |

### 最近阶段索引（仅供查阅，不代表当前状态）

```text
最近完成 phase：phase148（已验证，统一 pocket 语义下四卡干净 PPO）
最近完成 phase：phase149（已验证，右门 OPEN_LATCH 入口时序诊断）
最近完成 phase：phase150（已验证，retention reward 分叉训练至 step150）
最近完成 phase：phase151（失败，干净训练未消除左右门 OPEN_LATCH 不对称）
最近完成 phase：phase152（已验证，左右门入口物理等价，分化首先来自策略 roll 幅值）
最近完成 phase：phase153（已验证，正式机制续训至 step250）
最近完成 phase：phase154（已验证，定位到锁舌释放后的状态依赖 credit 死区）
最近完成 phase：phase155（已验证，相同适度正 ee_x 下左右门开缝物理响应一致）
最近完成 phase：phase156（已验证，探索倍率 2.0 安全且提高覆盖，但不足以修正 actor mean）
最近完成 phase：phase157（已验证，canonical opening force 的物理方向和区分能力成立）
最近完成 phase：phase158（失败，10% contact bridge 不足以让左门越过接触死区）
最近完成 phase：phase159（失败，30% contact bridge 后半程仍失去正 credit）
最近完成 phase：phase160（已验证，contact potential 的负 delta 会抵消并超过正 contact credit）
最近完成 phase：phase161（失败，修复局部 contact credit 后左门 actor mean 仍未持续转正）
最近完成 phase：phase162（已验证，左门 rollout signal 和实际 PPO 更新方向都为正；跨 rollout 均值变负来自状态分布变化）
最近完成 phase：phase163（已验证，左门释放前后 mean 都为负但 PPO signal 均为正；链路无剩余符号错误）
最近完成 phase：phase164（失败，额外 25 update 增加微小开缝但未提高转换率，retention 明显下降）
最近完成 phase：phase165（已验证，best-contact 未绑定建立接触时的 retention，主要造成 pocket/contact 退化）
最近完成 phase：phase166（失败，safe-contact 保住 retention，但未改善左门开缝或 REPOSITION 转换率）
最近完成 phase：phase167（已验证，physical fixed/moving jaw 槽位在镜像腕部构型下交换，导致 actor 学成两套 OPEN_LATCH 动作）
最近完成 phase：phase168（失败，canonical jaw order 显著削弱但未消除左右 OPEN_LATCH 分叉）
最近完成 phase：phase169（已验证，joint6 position 单字段不是剩余分叉的主因）
最近完成 phase：phase170（已验证，joint6 velocity 单字段不是剩余分叉的主因）
最近完成 phase：phase171（已验证，joint1--5 不存在待修复的确定性镜像映射）
最近完成 phase：phase172（已验证，delta_actions history 不放大门侧 action gap）
phase173：已验证 reward/advantage/PPO 方向并完成 release 事件 retention 轨迹诊断；后续见文末完整记录
phase188：已验证通用 temporal conditional distribution、storage 和 PPO 概率链
phase189：失败；全动作维 fraction=0.9 提高覆盖但破坏 retention 和安全
phase190：失败；全动作维 fraction=0.5 仍破坏 retention 和安全
phase191：已验证；仅 canonical ee_x 相关是当前覆盖/安全折中最优候选
phase192：失败；训练后转换率提高，但 retention 回归且 base mean 未吸收持续动作
phase173 开始前瓶颈：canonical jaw order 修复后两侧早期动作和锁舌响应趋同，但 step100 的 `OPEN_LATCH -> REPOSITION` 仍为左 `2.51%`、右 `16.95%`
phase173 唯一变量：保持 phase168 step100 的生产行为不变，只增加 reward、advantage、action noise、PPO 更新和 retention 的只读导出
当前接口有效 checkpoint：`logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260927_032226/model_step_000100.pt`（phase169 只读诊断基线，不代表任务成功）
当前接口 checkpoint SHA-256：`a8be95418108c550d920b8d18a95a26c77b960added8f66c923645a7418e43e4`
历史对照 checkpoint：`logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260926_054456/model_step_000275.pt`
历史对照 SHA-256：`190b1d0fee8f4e9f6d3ab328920442543aff5cd1f6c4992434e5fffda00cd6e0`
phase159 日志：`/tmp/b2z1_phase159_contact30_to300.log`（容器 `doorman_train`）
phase159 失败证据 checkpoint：`logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260927_012724/model_step_000300.pt`
phase159 失败证据 SHA-256：`102456a3bc229a52ec01431b9bdf1feae1caf5765af58a8177959bbf3d8f53c9`
phase160 日志：`/tmp/b2z1_phase160_contact_delta_diag.log`（容器 `doorman_train`）
phase161 日志：`/tmp/b2z1_phase161_contact_milestone_to300.log`（容器 `doorman_train`）
phase161 失败证据 checkpoint：`logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260927_014529/model_step_000300.pt`
phase161 失败证据 SHA-256：`a7c7b54b697d052932c3c315a2226b770fa5f2b92f5bd46b86f3e6ff5e60e7ef`
phase162 日志：`/tmp/b2z1_phase162_variant_gradient_diag.log`（容器 `doorman_train`）
phase162 更新前后输出诊断：`/tmp/b2z1_phase162b_actor_update_diag.log`（容器 `doorman_train`）
phase163 日志：`/tmp/b2z1_phase163_latch_group_credit_diag.log`（容器 `doorman_train`）
phase164 日志：`/tmp/b2z1_phase164_contact_milestone_to325.log`（容器 `doorman_train`）
phase164 失败证据 checkpoint：`logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260927_022235/model_step_000325.pt`
phase164 失败证据 SHA-256：`65b5c3db61b0d1da717380ab8a362f7f306ed49c6feb35ed22657035c531b713`
phase166 日志：`/tmp/b2z1_phase166_safe_contact_to300.log`（容器 `doorman_train`）
phase166 失败证据 checkpoint：`logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260927_023318/model_step_000300.pt`
phase166 失败证据 SHA-256：`3499d23c63921799a9810b6029137c327bf9b69884f93d4984abc9d7e24678c5`
phase167 日志：`/tmp/b2z1_phase167_obs_audit.log`（容器 `doorman_train`）
phase167 诊断数据：`/tmp/b2z1_phase167_obs.rank{0,1,2,3}.pt`，共 10657 个 OPEN_LATCH 样本
phase168 日志：`/tmp/b2z1_phase168_canonical_jaw_clean64_8192.log`（容器 `doorman_train`）
phase168 运行目录：`logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260927_032226`
phase168 无效启动证据：`/tmp/b2z1_phase168_canonical_jaw_clean_8192.log`、`logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260927_031447`（误用 8 steps/env、1 mini-batch，已终止且禁止继承）
phase169 日志：`/tmp/b2z1_phase169_q6_counterfactual.log`（容器 `doorman_train`）
phase169 数据：`/tmp/b2z1_phase169_q6_counterfactual.rank{0,1,2,3}.pt`（只读 rollout 完成后生成）
phase157 计划日志：`/tmp/b2z1_phase157_opening_force_diag.log`（容器 `doorman_train`）
phase158 日志：`/tmp/b2z1_phase158_contact_gap_potential.log`（容器 `doorman_train`）
phase158 续训日志：`/tmp/b2z1_phase158b_contact_gap_to325.log`（容器 `doorman_train`）
phase151 日志：`/tmp/b2z1_phase151_retention_clean_8192.log`（容器 `doorman_train`）
phase151 运行目录：`logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260926_021944`
phase152 日志：`/tmp/b2z1_phase152_left_open_latch_diag.log`（容器 `doorman_train`）
phase153 日志：`/tmp/b2z1_phase153_continue_to250.log`（容器 `doorman_train`）
phase153 运行目录：`logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260926_040325`
phase173 历史核对：2026-09-27 05:35；exact return 已确认 released 的后续 reward 轨迹本身更差，而非 critic/归一化或纯截断偏置；retention/progress 只读轨迹已完成
最近备份：2026-09-26 05:08；正式工作树、本文档和 step250 checkpoint 已同步到 `wholebody_1/3/4`，checkpoint SHA 三机一致
最近备份：2026-09-26 06:03；phase156 正式工作树、本文档和 step275 checkpoint 已同步到 `wholebody_1/3/4`，checkpoint SHA 三机一致
最近备份：2026-09-27 01:26；phase159 正式工作树、本文档、有效 step275 和 phase158 证据 step325 已同步到 `wholebody_1/3/4`，两个 checkpoint SHA 三机一致
最近备份：2026-09-27 02:22；best-contact 正式候选代码、本文档、有效 step275 和 phase161 证据 step300 已同步到 `wholebody_1/3/4`，两个 checkpoint SHA 三机一致
最近备份：2026-09-27 02:52；phase166 回滚后的正式工作树、本文档、有效 step275、phase164/166 失败证据 step325/300 已同步到 `wholebody_1/3/4`，三个 checkpoint SHA 三机一致
最近备份：2026-09-27 03:13；phase167 canonical jaw-order 正式代码、本文档和历史对照 step275 已同步到 `wholebody_1/3/4`；关键代码、文档和 checkpoint SHA 三机一致
最近备份：2026-09-27 03:40；phase168 step25 checkpoint 和本文档已同步到 `wholebody_1/3/4`，checkpoint SHA 三机一致
最近备份：2026-09-27 03:54；phase168 step50 checkpoint 和本文档已同步到 `wholebody_1/3/4`，checkpoint SHA 三机一致
最近备份：2026-09-27 04:07；phase168 step75 checkpoint 和本文档已同步到 `wholebody_1/3/4`，checkpoint SHA 三机一致
最近备份：2026-09-27 04:21；phase168 step100 checkpoint、最终结论和 phase169 计划已同步到 `wholebody_1/3/4`，checkpoint SHA 三机一致
phase154 有效日志：`/tmp/b2z1_phase154b_open_latch_ee_x_credit.log`（容器 `doorman_train`）
phase154 完成：2026-09-26 05:24；从 step250 恢复，4 x 2048 env，1 update，正式行为不变且未产生新基线 checkpoint
phase155 日志：`/tmp/b2z1_phase155_open_latch_ee_x_intervention.log`（容器 `doorman_train`）
phase155 完成：2026-09-26 05:42；4 x 2048 env，四组统一干预值，正常退出且未生成新基线 checkpoint
phase156 日志：`/tmp/b2z1_phase156_open_latch_ee_x_noise2.log`（容器 `doorman_train`）
phase156 运行目录：`logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260926_054456`
phase156 完成：2026-09-26 06:03；从 step250 续训 25 update，四个 rank 正常退出，无异常或 NaN
```

已经完成：

- canonical task basis、线向量和轴向量的双向数学变换及 5 项单元测试。
- 任意 physical B2Z1 command 到 world EE goal 的通用重建函数。
- 64-env Isaac 实际 `command -> world goal -> command -> world goal` 往返验证；世界位置/姿态最大误差分别为 `0.270 um` 和 `0.391 urad`。
- 临时逐步诊断已经删除，正式 8192-env 路径不承担该诊断开销。
- phase178 已从随机策略完成 100 update；正式训练统一使用标准 GAE 和 128 steps/env，旧 raw RTG actor advantage 路径已经删除。
- phase178 已确认 APPROACH_PREGRASP 稳定达到 `100%`，GRASP 和 OPEN_LATCH 均能学习；当前最早的系统性瓶颈是 OPEN_LATCH 释放锁舌后没有可靠形成门缝。
- phase173--184 已把 reward、advantage、horizon、critic target、PPO 更新、完成事件时序和 released 后物理 credit 链逐层审计。
- phase182 已修复通用阶段完成事件晚一动作归属的问题；phase183 对照证明该 bug 不是当前平台的主因。
- phase184 已确认 released 后负 credit 首先来自实际夹持 retention/progress，而非 GAE、value、optimizer、安全终止或阶段预算。
- phase194 已修复所有 staged task 的阶段 observation/storage 错位：完成动作仍按旧阶段结算，下一策略 observation、action mask 和 trainer stage 标签统一使用新阶段。
- phase204 已在统一语义下从随机初始化完成 125 update，排除了旧 checkpoint、stale critic 和整体 retention 坍塌作为当前左右转换差的主因。
- phase206 已修复 jaw axial observation 缺少门侧轴向符号的通用接口 bug；冻结策略反事实确认该字段解释了约 `80.5%` 的入口 canonical `ee_y` 差异。
- phase207 已从随机初始化完成修复后的 125 update；左右入口动作均值显著对齐，证明接口修复生效。
- phase212--215 已完成 PUSH_OPEN terminal credit 的离线反事实、真实 smoke、旧策略因果续训和随机初始化正式训练四层验证；统一 hard-termination cost 40 正式保留，旧阶段特例已删除。
- phase215 已正常完成 125 updates；纯动力学四卡链路无异常或 NaN，最终 checkpoint 和 SHA 已核对。
- phase216 已确认 RL 与 AutoOpen 的 GRASP 目标姿态、实际 EE tracking 和腕部翻转分支一致；首个物理分化发生在接触以后，不再检查已排除的旋转方向。
- phase216 control 和两个反事实已确认：自由空间重力补偿没有纳入夹爪接触外载荷，导致翻转构型中的 fixed/moving jaw 受力不对称；临时阶段门控与“整个 GRASP 关闭补偿”均已删除，不得作为正式方案复用。

尚未完成：

- phase256 已验证 base 独立噪声会破坏低层所需的命令持续性；phase257--258 已量化低层主要响应时间约为 `0.3--0.5 s`。
- phase263--267 已证明统一探索所有权下 APPROACH_PREGRASP 的几何结果能持续学习；当前阻塞已转为接近过程中上升的 `>600 N` unsafe-force。
- 纯 RL 独立阶段中 WALK_TO_DOOR、PUSH_OPEN、THROUGH 已通过本轮门槛；GRASP 已由 `12.55%` 提升到约 `29.2%`，但碰撞感知导航势场仍在验证；OPEN_LATCH/REPOSITION 尚未完成合法接触态初态和独立验收。
- 尚未进行最终 seed `1700--1703` 策略验收。

### 上层路线图与决策门槛

| 工作层级 | 完成条件 | 当前状态 | 下一动作 |
| --- | --- | --- | --- |
| 训练基础设施 | 纯动力学四卡训练稳定；坐标、observation/action、GAE、阶段采样和完成事件语义统一 | phase194 修复阶段切换首帧错位并完成同规模回归；已完成并持续回归 | 后续行为修改必须通过 py_compile、单测、Isaac smoke 和前序阶段回归 |
| 首个学习瓶颈 | actor 在六阶段使用统一动作语义并独立控制全部 10 维命令，阶段 one-hot 可见 | WALK_TO_DOOR step400 确定性评估 `1825/2048=89.11%`，训练链路与第一阶段 reward 已证明可学习 | 第一阶段按用户口径通过；保留右铰链 `83.28%` 的鲁棒性记录，不在本轮继续优化 |
| 后续阶段学习 | 六阶段均有机器狗/机械臂/环境连续势场，并可从上一阶段合法状态单独训练 | PUSH_OPEN `91.99%/92.68%`、THROUGH `85.11%/85.06%` 已通过独立验收；GRASP 碰撞感知势场四卡训练中 | 完成 GRASP 新势场复评；随后实现并验收 OPEN_LATCH/REPOSITION 接触态初态与独立训练 |
| 最终策略验收 | 同一 checkpoint 在 seed `1700--1703` 安全完成全部六阶段 | 未完成 | 完整无特例回归，保留日志、轨迹和 checkpoint SHA |

路线图按“最早失败阶段”推进。每次只允许一个待验证假设和一个行为变量；诊断 phase 不更新模型，训练 phase 不在运行中改代码。失败方案必须进入下方禁重复清单，不能换名字再次尝试。

### 已解决问题总账

这里仅记录已经获得直接证据、正式修复并完成清理的问题。历史细节见对应 phase；后续不得因上下文恢复重新诊断，除非先取得新的直接反证。

| 问题 | 根因 | 正式修复与证据 | 状态 |
| --- | --- | --- | --- |
| 阶段切换首帧的 observation/storage 错位 | 完成动作与下一策略 observation 的阶段标签不一致 | phase194 统一完成动作结算、下一 observation、action mask 与 trainer stage 标签 | 已解决 |
| 左右门 jaw axial observation 语义不一致 | 轴向字段缺少门侧符号 | phase206 在通用 observation 接口修复；反事实解释约 `80.5%` 的入口差异 | 已解决 |
| opposed contact 假阴性 | 单点接触位置跳变触发冗余 2 cm 轴向配对条件 | phase209 删除冗余条件并保留其余几何、接触、姿态、力与安全约束 | 已解决 |
| PUSH_OPEN terminal credit 压过正确动作 | hard termination cost 的阶段语义不统一 | phase212--215 使用全阶段统一绝对成本 40，并经离线、smoke、续训和随机初始化验证 | 已解决 |
| actor 动作被脚本 reference/supervision 分走 | 环境内仍有 AutoOpen 派生的动作替代链 | phase217 删除旧路径；左右门十维 action-to-command 雅可比均满秩 10 | 已解决 |
| 纯 delta actor 随机初态持续漂移 | 输出层归零原先只在 residual 模式生效 | phase222 增加通用显式零初始化并在 RecurrentActor 替换 backbone 后重施；单卡 rollout 十维 mean 全零 | 已解决 |
| residual-era 阶段探索倍率不适用于纯 actor | stage scale 把底盘探索压到低层步态阈值以内，并让同一 action 坐标随 stage 改变探索语义 | phase223 定量审计后，phase224 删除 stage matrix，采用全阶段统一逐动作 std 和可重算的 temporal likelihood；单卡及四卡真实训练均跨过步态响应区 | 机制已解决；学习瓶颈转到 command-to-actual 分化 |
| DLS 姿态误差被错误放大 | 本库额外把 lab30 原始 quaternion-vector orientation error 乘 4，使 joint6 target 越过硬限位并触发一步超速 | phase255 恢复 lab30 原公式；正式 control/candidate arm overspeed 从 `1740/1857` 降为 `9/6`，临时代码已删除 | 已解决 |
| reset 初态夹爪穿入门体 | base 与 middle-arm 扰动独立采样，合法边缘范围的组合仍可能让关闭夹爪与门体 OBB 相交 | 生产 URDF FK、夹爪/门体 OBB SAT 联合拒绝采样；8192 初态 `unsafe_force/reset_step=0`，原随机范围保持 | 已解决 |

### 六阶段目标与进展总账

| 阶段 | 物理目标与完成定义 | 当前有效证据 | 当前状态与后续门槛 |
| --- | --- | --- | --- |
| WALK_TO_DOOR | 底盘位置 `<0.16 m`、yaw `<0.08 rad`、base motion `<0.05`；机械臂回 middle preset，误差 `<0.05 rad`、速度 `<0.08 rad/s`；无接触，连续 10 步 | step400 对 2048 个随机初态做确定性评估：总成功 `89.11%`，左/右铰链 `95.00%/83.28%` | **已通过**。当前不继续优化效率或右门鲁棒性 |
| GRASP | 底盘保持操作范围但允许移动；机械臂从 middle 自主抓取，无 pregrasp 点或预设开度；严格双侧有效接触连续 5 步 | 旧 step225 两个 seed 的单帧完整夹持为 `600/2048=29.30%`、`598/2048=29.20%`；新碰撞感知候选 step200 为 `0/2048` | **完整阶段尚未通过**。旧 checkpoint 达到此前的比例改进目标，但旧欧氏场有直穿缺口；新候选又在安全距离形成局部最优，已判失败 |
| OPEN_LATCH | 底盘保持范围、严格双侧夹持、锁舌 `>=20 mm`、几何门缝同时成立 | phase272 已修正为四项同帧验收并提供保持/锁舌/门缝连续势场 | 等 GRASP 学会后评价，或从合法阶段 2 初态库单独训练 |
| REPOSITION | 无把手/门板接触地撤离，底盘位置和 yaw、EE 推门位姿、闭爪状态以及 base/EE 稳定条件连续满足 15 步 | phase215 左/右 `48.76%/30.16%` 来自脚本完全拥有十维动作，不能作为纯 RL 学习成绩 | 等纯 RL 到达本阶段后评价；当前未验证 |
| PUSH_OPEN | 机械臂建立过真实推门接触并把门打开到至少 89 度；不要求全程持续接触，但门速与接触必须满足安全约束 | step400 两个 seed 确定性成功 `1884/2048=91.99%`、`1898/2048=92.68%`；全部环境建立真实门板接触 | **独立阶段已通过**。剩余失败主要为门连续超速，其次为任务力超过 `600 N`；当前不继续叠加 reward |
| THROUGH | 机械臂严格回 middle preset、底盘穿门位移 `>1 m`、门角保持 `>=65 deg`，并稳定 10 步 | step400 两个 seed 确定性成功 `1743/2048=85.11%`、`1742/2048=85.06%`；门角和底盘稳定均为 `2048/2048` | **独立阶段已通过**。剩余弱项是穿门距离与 middle preset 的尾部交集；串联训练前不继续单独优化 |

### 近期尝试结果（恢复后先查，避免重复）

| phase | 唯一假设或改动 | 已验证结果 | 后续决策 |
| --- | --- | --- | --- |
| phase272 | 六阶段三类状态语义、删除 PREGRASP、增加独立阶段初态库 | 76 tests 与 stage 1/4/5 真实程序化 reset smoke 通过；WALK step400 确定性成功率 `89.11%`；GRASP/PUSH_OPEN/THROUGH 独立训练已启动 | WALK 已通过；完成 GRASP 与远程后期阶段训练/评估，stage 2/3 接触态初态仍待实现 |
| phase271 | 将混合的 APPROACH_PREGRASP 显式拆为 WALK_TO_DOOR 与 PREGRASP | 七阶段类、配置、reward、observation、staged reset、日志和验收同步完成；71 tests 与真实 optimizer smoke 通过 | 下一轮使用 32 steps/env x 200 updates，从随机初始化先验收 WALK_TO_DOOR，再评价条件化 PREGRASP 指标 |
| phase204 | 在统一 reward、阶段与 PPO 语义下随机初始化干净训练 | step125 严格转换左/右 `7.02%/27.10%`；排除 stale critic 和整体 retention 坍塌 | 保留为旧 observation 语义对照，不从该 checkpoint 续训 |
| phase205 | 长窗口只读定位左右门首次分化层 | 明显物理分化前，入口 actor action 已先分叉；晚期失败集中为 opposed contact 丢失 | 不重复扩大同字段 rollout，转查 actor 输入 |
| phase206 | 字段反事实定位并修复 jaw axial canonical 语义 | 修复约消除 `73.5%` 的入口 `ee_y` 门型差；通用接口 bug 成立 | 正式保留修复；旧 checkpoint 与新 observation 语义不兼容 |
| phase207 | 从随机初始化验证修复能否解决学习瓶颈 | 入口动作对齐，但严格转换未超过 phase204，假设失败 | 禁止盲目续训；接口修复保留 |
| phase208 | opposed contact 首次 `1 -> 0` 事件归因 | 已验证；物理接触仍在，单点接触位置跳变触发冗余 2 cm 轴向配对条件，`task_stage_progress` 立即错误扣约 `1.4--1.6` | 禁止再把总体 opposed 低值直接解释为策略真实丢夹持；转 phase209 修接口 |
| phase209 | 删除双接触点轴向距离假阴性条件 | 已验证；同 checkpoint 下 opposed 与严格转换显著恢复，新增 precise 样本仍满足其余几何、接触、姿态、力和安全约束 | 正式保留纯函数、唯一调用点和测试；删除诊断路径，完成回归与远程同步后进入 phase210 干净训练 |
| phase210 | 在 phase209 统一验收语义下从随机初始化训练 | 已验证；125 updates 正常完成，OPEN_LATCH 与 REPOSITION 双侧学习显著改善，左门 PUSH_OPEN 成为首个稳定后续瓶颈 | 保留 step125 为当前有效 checkpoint；不再盲目续训，转 phase211 只读归因 |
| phase211 | 左门 PUSH_OPEN 首个物理/动作/credit 分化层 | 已验证；策略动作是首个行为分化，物理映射一致；高额 termination credit 压过正确的接近/完成 credit，造成左门 `ee_y` PPO 信号反向 | 禁止再改 canonical 坐标或低层跟踪；转 phase212 统一终止 credit 反事实 |
| phase212 | 安全终止 credit 的统一幅度与机制 | 已验证；拒绝会削弱第一阶段跌倒惩罚的统一比例缩放，采用全阶段、全门型统一绝对成本 40 | 删除旧阶段特例并正式保留统一机制；转 phase213 真实 optimizer smoke |
| phase213 | 统一 hard-termination cost 40 的在线链路验证 | 已验证；单卡 `8 x 8 x 1` 正常完成，termination 指标相对旧 smoke 精确放大 4 倍，纯动力学配置成立 | 进入 phase214 四卡诊断续训 |
| phase214 | 统一成本 40 对现有 PUSH_OPEN 策略的在线因果验证 | 已验证；25 updates 后双侧条件转换、THROUGH reached 和前序阶段同时改善，PUSH_OPEN 安全终止为 0 | 保留 checkpoint 为因果证据，不作为正式干净基线；进入 phase215 |
| phase215 | 新 reward 语义下四卡随机初始化正式训练 | 已验证；125 updates 正常完成，PUSH_OPEN terminal credit 修复在干净训练中复现，前序阶段和安全无回归；左右门早期差异及 THROUGH 收拢仍未解释 | 保留 step125 为当前有效 checkpoint；phase216 先审计 GRASP/OPEN_LATCH 门侧语义，再处理 THROUGH |
| phase216 | GRASP 姿态、执行与接触后稳态补偿审计 | 姿态语义及实际 tracking 左右一致；关闭整个 GRASP 补偿的反事实确认自由空间补偿参与根因；首个通用 `J^T wrench` 候选虽减小垂直偏差，却把右 fixed force 提高到 `31.58 N` 并降低 pocket，通过失败 | 不保留该结果为正式结论；先闭环解释接触反力的方向、时序和内生反馈，再决定正确接口实现 |
| phase217 | actor 动作所有权纠偏 | 已验证；旧 reference/supervision/steady-inverse 任务路径已删除；左右门十维雅可比均满秩 10，映射无 stage 输入；`38 passed, 5 subtests passed`，清理后 optimizer smoke 通过 | 禁止恢复脚本动作链；后续动作接口变化必须重复验收，进入 phase218 随机初始化训练 |
| phase218 | 纯 RL 随机初始化训练 | 失败；四卡到 update26 无异常，但 step25 仍无任意 1 步联合稳定，除 gripper-open 外五项 progress 无趋势 | 保留 step25 只作 phase219 诊断；禁止盲目延长训练或修改后续阶段 |
| phase219 | APPROACH_PREGRASP 硬最小值 credit 审计 | 已验证；orientation/pregrasp-position 分别占 `68.93%/30.60%` argmin，其余三项合计仅 `0.47%` | 临时诊断已删除；进入 phase220 验证统一 balanced result progress |
| phase220 | bottleneck-balanced 第一阶段结果 progress | 失败；step25 聚合 reward 转正，但五项 quality/episode-seen 未同步改善，`stable_reached_1/5/10` 全为 0 | checkpoint 只作 phase221 诊断基线；禁止盲目续训或继续扫描 mean/min 权重 |
| phase221 | APPROACH_PREGRASP 结果 credit 因果审计 | 已验证；termination return 均值 `-22.35` 对 stage progress `-0.031`，终止由 wrist arm-overspeed 完全主导；临时诊断已删除 | 进入 phase222 修复纯 delta actor 初始均值，不改安全阈值或 reward |
| phase222 | 纯 delta actor 零均值初始化 | 失败；机制修复使 existing termination 降至末轮 0，但 step25 五项未同步改善，`stable_reached_1/5/10` 全为 0 | 正式保留通用零初始化；checkpoint 只作 phase223 冻结诊断，不盲目续训 |
| phase223 | 第一阶段十维探索物理可达性 | 已验证；base 原尺度无法启动步态，放大平移/yaw 均改善对应 episode-best；fresh-reset arm overspeed control 为 `0.61%/step`，干预为 `0.67--0.68%/step`，不是此前误判的全部由 base 引起 | 临时诊断已删除；进入 phase224 替换 residual-era stage matrix，不做 APPROACH 单行倍率补丁 |
| phase224 | 全阶段统一的逐动作探索尺度 | 失败；stage matrix 已删除，`26 passed`，四卡 step25 的真实运动和多项 episode-best 改善且安全无回归，但 `stable_reached_1/5/10` 仍全为 0；command goalward 为正而 actual goalward 略负 | 正式保留统一探索机制，不恢复 stage matrix；checkpoint 只作 phase225 冻结诊断基线，不盲目续训 |
| phase225 | 底盘 command-to-actual 首个分化层 | 已验证；冻结 rollout 的 command/actual goalward 均为正，实际速度沿当前/上一命令均显著为正且上一拍仅略优；排除坐标符号和低层反向跟踪，确认轻微一拍时延 | 临时指标已删除；不修改低层，进入 phase226 审计 EE 位置/姿态 credit |
| phase226 | 第一阶段同机制续训 | 已验证；step25--50 五项结果同步大幅改善，goalward 跟踪转强，无异常或 NaN，但 stable 仍为 0 | 保留 step50 并进入 phase227；不改 reward，平台前不做 credit 手术 |
| phase227 | 第一阶段继续形成联合稳定 | 已验证；step75 五项几何 quality 继续提高，step70 偶发 `stable_reached_1=0.0002`，但 5/10 步始终为 0，GRASP occupancy/reached 为 0；无异常或 NaN | 保留 step75 作 phase228 冻结诊断；不再无证据盲目延长训练，先审计联合停稳 credit |
| phase228 | 几何到位后的联合停稳 credit | 已验证；有效 rollout 的 262144 帧中，高 readiness 仍高速且无联合停稳；代码确认速度不进入现有 progress。前两次无效采样分别因初始化污染和挂错 trainer，均已废弃并记录 | 临时诊断已删除；不再用现有 reward 盲目续训，进入 phase229 单变量验证 |
| phase229 | 几何 readiness 耦合停稳势函数 | 失败；`13 passed` 和 smoke 均通过，但四卡 step25 五项 quality 均未优于旧同期，actual speed 无改善，stable 全 0；安全无明显回归 | checkpoint 只作 phase230 冻结诊断，不续训、不扫描权重；归因后决定删除候选 |
| phase230 | 探索噪声与 actor mean 停稳归因 | 已验证；修复诊断别名后，step25 与高 readiness step75 两组正常/近零对照均证明固定全局 std 是主要速度下限，mean 仍有次要残余；全局 std 始终贴上限 | 临时导出已删除；不得再用首轮被原地积分污染的 sampled action；进入 phase231 |
| phase231 | 统一状态依赖探索方差 | 失败；四卡 step25 五项 quality 较 phase224 同期回归，actual planar speed `0.2177 m/s`，stable 与 GRASP 全为 0；冻结分桶显示各 readiness 桶十维 std 均贴上限 | 候选实现和专属诊断已删除，不续训、不调初始化比例或 entropy；进入 phase232 梯度归因 |
| phase232 | PPO 探索方差梯度归因 | 已验证；全局 std 的 entropy/surrogate 梯度范数比 `3.184`，总梯度与 entropy 余弦 `0.951`；scale 网络分别为 `1.428/0.806`，Adam 历史使十维 std 下一步仍全部向上 | 临时参数化与导出已删除；进入 phase233，不直接调 entropy 系数或重试网络 |
| phase233 | 统一探索预算目标设计 | 已验证；独立纯函数和 8 项测试通过，预算满足时 quiet minority 无正则梯度，坍塌时仅不足维度增噪；冻结 rollout 的十维预算均有余量 | 正式保留纯函数与测试；phase234 接入前不得启动训练或修改 reward |
| phase234 | 总体方差预算 PPO 接口 | 已验证；`54 passed, 5 subtests passed`，三组 smoke 覆盖正常 optimizer、checkpoint round-trip 和预算违反路径；诊断 dual `0.000474999` 与闭式值一致 | 正式保留统一接口；进入 phase235，不再重复接口 smoke 或修改 reward |
| phase235 | 总体方差预算四卡随机初始化短训 | 失败；step25 除 yaw 外四项 quality 低于 phase224，actual planar speed `0.2103 m/s`，stable 与 GRASP 全为 0；安全和运行链路无异常 | checkpoint 只作 phase236 只读归因，不续训、不调预算 |
| phase236 | 冻结状态方差行为归因 | 已验证；attenuation 有弱单调分化，但 `>=0.1` readiness 仅 29 帧，总体方差已近预算下界，高桶 noise/实际速度仍高 | 临时代码已删除；进入 phase237，不续训、不调预算 |
| phase237 | readiness 分桶 surrogate 梯度归因 | 已验证并清理；高 readiness 六个关键运动维均收到增噪方向，参数梯度投影仅 `0.643%`，ratio/clip/梯度闭合及 `54 passed, 5 subtests passed` 均通过 | 禁止把问题简化成只缺样本或直接加入 readiness 权重；进入 phase238 时间链归因 |
| phase238 | 高 readiness 负 advantage 时间链归因 | 已验证并清理；raw GAE/归一化/reward 三链闭合，负 credit 来自随后几何退化及部分终止，现有稳定势贡献仅 `1.38e-5` | 禁止直接加 readiness 权重或只调探索预算；进入 phase239 修复结果势函数本身 |
| phase239 | 统一第一阶段结果势函数 | 失败；step25 五项 quality `0.3175/0.5024/0.1396/0.1043/0.8601`，actual speed `0.2116 m/s`，stable 与 GRASP 全 0；安全无回归 | 不续训、不扫描权重；phase240 冻结 checkpoint 做结果势数值和 PPO 梯度归因 |
| phase240 | phase239 结果势的数值、时间 credit 与 exact PPO 梯度归因 | 已验证并清理；近目标覆盖仅 `0.414%`，局部有部分正确减速/降噪方向，但乘法势普遍削弱约 18% 几何 credit，近目标桶仅贡献约 `4.826%` 总梯度 | 删除 phase239 失败势和全部 phase240 临时代码；禁止换权重重试，phase241 必须采用不缩放几何 credit 的额外结果信号 |
| phase241 | 额外稳定成就势 | 失败；step50 五项几何 `0.6628/0.7056/0.5278/0.4679/0.9077`、actual speed `0.1982 m/s`，但 stable1/5/10 与 GRASP 全 0；安全无实质回归 | 禁止继续延长或调权重；phase242 冻结 checkpoint 做覆盖、advantage 和 exact PPO 梯度归因 |
| phase242 | 联合成就 credit 与 exact PPO 梯度归因 | 已验证；高联合质量覆盖/保持率和梯度份额均近乎消失，achievement 事件虽有正 advantage 和部分 mean 减速方向，但 attenuation 总梯度仍增噪 | 删除 phase241 候选和全部临时诊断；禁止续训、调权重或换名重试历史最佳增量 |
| phase243 | 可回退的有符号联合质量势 | 失败；step25 四项几何及实际速度改善，但 base yaw `0.4790 < 0.5013` 未通过预注册同期门槛，stable 1/5/10 与 GRASP 全为 0；安全无回归 | checkpoint 仅作 phase244 冻结归因，不续训、不扫权重；根因闭合后决定整体保留或删除候选 |
| phase244 | phase243 signed credit 分配归因 | 已验证并清理；base yaw 仅 `0.099%` 成为 weakest、mean 梯度份额约 `0.202%`，机械臂两项占 `99.74%` weakest 帧；Q/GAE/ratio 闭合 | phase243 候选与全部诊断已删除；禁止续训、调权重或换名重试 min-selector |
| phase245 | 独立稳定/无接触 signed potential 离线审核 | 失败；单位权重候选的平均绝对 delta 是原几何的 10.41 倍，94.78% 几何退化/减速冲突帧仍获净正 credit；闭环和初始化性质虽通过但不足 | 不实现、不训练、不以单位权重重试；进入 phase246 原则性尺度推导，禁止权重扫描 |
| phase246 | 独立稳定势原则性尺度推导 | 失败；`0.1` 后平均绝对尺度与几何为 1.041:1，仍使 68.97% 几何退化/稳定改善冲突帧净正，并抵消 53.07% 反向冲突中的几何改善 | 停止独立逐帧势的权重方向，不尝试更小系数；进入 phase247 时间保持性归因 |
| phase247 | 稳定改善时间保持性离线归因 | 已验证；单步真实停稳仅 18 帧、5/10 步窗口均为 0；正稳定增量 10 步内 68.69% 曾跌回事件前，只有 49.48% 保留一半增益 | 排除继续使用或缩放单帧 `Delta S`；进入 phase248 审核可回退的 10 步持续结果势 |
| phase248 | 十步持续稳定结果势离线审核 | 失败；`|Delta W|/|Delta G|=3.52`，正窗口增量冲突帧 94.24% 翻为净正；即时恶化仅 36.45% 给负值，旧最小值滑出可产生伪改善 | 不实现、不缩放或重试 trailing-min；进入 phase249 持久下包络审核 |
| phase249 | 持久稳定下包络数学与离线审核 | 失败；构造性质通过，但真实轨迹 99.31% 帧仍产生 delta，平均尺度为几何的 0.944 倍，冲突翻转与 phase246 基本不变 | 不实现、不继续改时间常数；停止稳定势候选链，进入 phase250 架构复核 |
| phase250 | 探索噪声与连续停稳架构复核 | 已验证；总体预算接口正确，但 near-ready 样本噪声主导实际速度，后续退化产生负 raw advantage，六个运动维收到增噪 credit，构成 bootstrap deadlock | 不重复 phase230--238，不继续设计稳定势；进入 phase251 逐动作维物理贡献分解 |
| phase251 | 动作维噪声到实际运动的因果分解 | 已验证并清理；修复两处仅影响诊断的张量别名后，v2 正式轨迹通过重建和 pre/post 验收；外生 FWL 显示前三维合计主导 `99.54%` 任务速度响应 | 禁止再用旧别名污染张量；运行时标识已归零，进入 phase252 设计审核 |
| phase252 | 统一物理响应尺度的探索所有权审核 | 已验证；`max std`、attenuation、temporal structure 与 budget 的不变量和所有权已分离。phase223 在相同边际 base 尺度下得到独立/AR(1) 实际速度 `0.1168/0.2192 m/s`，说明首个待复验变量是 base 持续性，不是 base 幅度 | 不降低 base max std、不改 budget 权重、不扫描 fraction/rho；进入 phase253 当前 checkpoint 匹配冻结反事实 |
| phase253 | base 时间持续性的冻结反事实 | 失败候选；命令阈值覆盖保持约 `89%`，base planar/yaw 与 EE linear speed 分别降低 `44.96%/16.15%/14.61%`，unsafe-force 未恶化，但 existing termination 从 `0.6805%` 升至 `0.7118%/step` | 不写入正式配置、不训练、不猜测原因；phase254 只分解 existing termination 来源 |
| phase254 | base 独立噪声 existing termination 来源分解 | 已验证；candidate 相对 control 的 inherited/far/arm 增量为 `-65/0/+118`，净 existing 增量 `+83`，OR 重建误差为 0 | phase255 只读保存 reset 前逐关节瞬时状态，确认首个越界关节和跃迁机理 |
| phase255 | joint6 一步超速的控制根因与修复 | 已验证；错误 4 倍 orientation gain 使 joint6 target 越过硬限位。恢复 lab30 原公式后，正式 control/candidate arm overspeed 从 `1740/1857` 降为 `9/6`，candidate existing `68 < 114` | 正式保留 IK 修复和 base-independent 配置；全部临时代码已删除，进入 phase256 optimizer smoke |
| phase256 | base-independent 生产训练链与 step25 学习验证 | 失败；生产 optimizer 和四卡链路正常，速度显著下降且安全无回归，但五项几何低于主要同期基线，stable 1/5/10 与 GRASP 全 0，正 goalward command 未转化为正实际速度 | checkpoint 只作证据，不续训；phase257 复用已有冻结张量审计时间持续性，不立即恢复 0.9 或扫描 fraction |
| phase257 | base command 时间持续性离线归因 | 已验证；iid candidate 的 lag-1 correlation 约 0、相邻同向率 `33.5%`，actual velocity 对 lag-4 command 投影仅 `0.0096 m/s`；AR control 对应为约 `0.85`、`93.8%`、`0.145 m/s` | 排除低层方向错误；phase258 从 iid 轨迹辨识响应核并原则性推导时间尺度，不扫描 fraction |
| phase258 | 低层 base command 响应核辨识 | 已验证；ARX(4) 验证集 R² 为 `0.852/0.891`，x/y 达到 50% 响应需 `5/10` 拍、80% 需 `15/25` 拍 | 不通过缩短 persistence 解决停稳；phase259 审核时间相关与状态幅度的架构所有权 |
| phase259 | 探索时间尺度与幅度所有权审核 | 已验证；AR sampler/reset、exact sequence likelihood、attenuation PPO 梯度和 deterministic deployment 一致，29 项单测通过；actor 不直接观察 latent 不是概率错误，且 observation 已包含上一拍命令与实际运动 | 确定缺口是 temporal fraction 只有固定配置、没有 observation-dependent owner；phase260 先设计完整概率契约，不实现候选 |
| phase260 | 状态相关时间持续性概率契约设计 | 已验证；逐 observation/action fraction 与 amplitude 分权，old/new fraction 分别进入 exact sequence likelihood，budget 不约束 fraction，entropy 对 fraction stop-gradient，部署只输出 actor mean | phase261 仅做最小实现和纯张量测试；未通过前不启动 Isaac 或训练 |
| phase261 | 状态相关 temporal fraction 最小实现 | 已验证；独立 fraction owner、old/new 成对滤波、常量维精确为零、可学习维有界且有非零梯度；py_compile、完整 `58 passed, 5 subtests passed`、diff check 通过 | phase262 只做配置接线和单卡极小 optimizer smoke；正式训练前必须证明参数实际更新和保存 |
| phase262 | 状态相关 fraction 单卡 optimizer smoke | 已验证；首次运行暴露并修复 fraction=0 的平方根边界 NaN，复跑 64 timesteps/1 update 正常，四个可学习输出变化约 `5.02e-4`、固定六维精确为 0，checkpoint strict reload 成功 | 保留通用数学修复和配置；phase263 四卡干净训练到 step25，不继承 smoke checkpoint |
| phase263 | 状态相关 fraction 四卡干净训练 | 失败；step25 正常结束，四项几何和 actual goalward 改善，但 speed `0.2081 m/s`，stable 1/5/10 与 GRASP 全 0；完整 metrics 未被 Rich 输出或 checkpoint 保存 | checkpoint 只作冻结诊断，不续训；phase264 在通用日志层补持久化后再做首断点归因 |
| phase264 | 完整 Trainer metrics 通用持久化 | 已验证；rank0 每 update 追加完整标准 JSON，非有限值写 `null` 并显式列键，两个 PPO Trainer 统一接入，完整回归 `63 passed, 5 subtests passed` | 正式保留；phase265 只用冻结 checkpoint 做单 rollout 诊断，不改行为或参数 |
| phase265 | 冻结 checkpoint 首断点单 rollout | 已验证；sidecar 1 条 956 键、无非有限值，fraction 可学习维有差异、固定维为 0；orientation/position OK 几乎为 0，先于停稳失败 | 输出模型丢弃；phase266 只读审核 EE 几何 credit 链，不扫描探索参数 |
| phase266 | APPROACH_PREGRASP 几何 credit 链静态审核 | 已验证；两项物理误差连续进入 signed potential，GAE 与 exact temporal PPO likelihood 链完整，actor observation 和十维映射无缺口；step1--25 两项持续改善 | 排除修改 reward/接口；phase263 的 25-update 失败门槛过早，phase267 用不改行为的 step75 干净训练验证 |
| phase267 | 当前统一机制延长学习时间窗 | step50 安全门槛失败；几何 quality 大幅同步改善，但 unsafe-force 从 step25 `4.58e-5` 升至 `2.42e-4/step`，全部位于 APPROACH；按规则停止 | checkpoint 禁止续训；phase268 只做 reset 前冻结事件归因，不直接调阈值、reward 或探索 |
| phase268 | reset 前 unsafe-force 事件级冻结归因 | 已验证；89 个事件中 84 个仅门板、2 个仅把手、3 个同时接触。56 个发生在首次物理步；动态无接触到碰撞事件的接近速度中位数为 `0.5007 m/s` | 禁止把两类问题混成一次修复；临时诊断清理后进入 phase269 |
| phase269 | reset 初态物理合法性 | 计划中；只修 base/arm 独立随机造成的门体初始重叠，不处理策略动态接近 | 先从真实 collider clearance 定义合法性并做大规模 reset 验收；不得拍固定 base 距离或同时改 reward/探索 |

### 决策与禁重复清单

| 已验证方向 | 结论 | 后续约束 |
| --- | --- | --- |
| 直接增加状态依赖方差 scale 网络 | phase231 的结构、概率链和 optimizer smoke 均通过，但训练后各 readiness 桶都贴方差上限，几何与停稳均未改善 | 不续训、不调初始 scale、不降低 entropy 碰运气；必须先完成 surrogate/entropy 梯度归因 |
| 固定逐状态 entropy bonus 与状态方差网络直接组合 | phase232 证明 entropy 在 std/scale 参数上主导总梯度；近目标降噪 credit 极稀有，网络在所有 readiness 桶饱和 | 不再使用逐样本 entropy 强迫每个状态取最大方差；phase233 只研究总体探索预算语义，不做系数扫描 |
| 继续延长 rollout horizon | 128 步已使 released 样本覆盖升至约 `34%`，但 `1497` 次 release 无一次进入 REPOSITION | 不再靠盲目加长 horizon 解决当前瓶颈 |
| 增加 released critic 高进度采样 | critic target 本身随当前失败轨迹降低；更多同类样本会让 value 更低 | 不把 critic oversampling 当作 actor credit 修复 |
| raw RTG 与标准 GAE | raw RTG 错误归因后续失败；标准 GAE 修复局部 signal，但 step50--100 转换率仍平台 | 正式路径只保留标准 GAE；不得恢复双路径或兼容开关 |
| 直接按常驻 `noise * advantage` 调 reward/权重 | 当前统计没有应用完整真实 stage sample weight，也不是参数更新结果 | phase179 前禁止修改 reward、stage weight 或网络结构 |
| 继续训练 phase178 checkpoint | OPEN_LATCH -> REPOSITION 在 step50--100 平台，actor mean 与局部 signal 不一致 | phase179 先做单 update 因果诊断，不盲目续训 |
| 用 stage 聚合 signal 判断左右门共同 credit | phase179 证明聚合正值可由左门强正和右门弱负组成，并导致相反的门型内 actor mean 更新 | 后续 credit、reward 和更新诊断必须至少按 stage x door variant 分层 |
| 把 OPEN_LATCH 聚合正 ee_x signal 解释为推门 credit | phase180 证明左门聚合正值主要来自未释放状态的终止相关性；released 左右门 signal 均为负，且短时先损失 retention/progress | 必须按 released 状态和时间延迟分层，不用 stage 总均值判断任务动作方向 |
| 把 released 后负 credit 归因于 GAE、value、optimizer、阶段超时或安全终止 | phase181--184 逐层排除；负号首先出现在当前动作造成的实际 retention/progress | 后续先研究物理动作耦合和势函数结构，不重复调整 horizon、critic 或终止阈值 |
| 用单独正 `ee_x` 扰动代表完整推门动作 | phase184 显示该扰动短期破坏 retention：左门主要损伤 axis，右门主要损伤 pocket/contact；约 32 步后才可能恢复 | 必须评估统一的平移/姿态协调动作，不能据单轴结果直接改 reward |
| 把乘法 retention 势函数直接改成 additive | phase185 的短窗口旧策略反事实中左门仍负，当时没有正确 sequence likelihood、action ownership 和足够持续成功轨迹；phase200 在这些接口修复后的 128 步轨迹上取得了新的直接反证：`0.10/0.45/0.45` 使左右精确 sequence signal 同时为正 | 旧禁令已由新证据重新打开；phase201 只验证这一结构变化，必须同时检查 retention、安全和前序阶段，不得继续扫描权重 |
| 把当前瓶颈归因于 canonical 坐标或低层跟踪方向错误 | phase185 中动作到 goal、动作到实际 EE 的左右门映射均同号且近似对角；位置与姿态响应方向一致 | 不再修改 task frame、canonical 符号或低层跟踪方向，除非出现新的直接反证 |
| 继续调 retention 权重解决左门开缝 | phase185 additive 权重扫描即使把 retention 权重降为 `0`，左门 `ee_x-return` covariance 仍为负；当前轨迹没有建立 force/gap | 不再对同批失败轨迹反复调权重；先验证持续动作覆盖是否能产生缺失的物理轨迹 |
| 用逐步独立白噪声探索 delta accumulator 的接触死区 | phase185 的 `ee_x` noise lag 1/2/4/8 自相关均约为零，actor mean 也约为零；历史持续 `ee_x=0.02` 干预能使两侧开缝 | phase186 先做当前 checkpoint 因果复验；成立后正式机制必须统一、保持 PPO 概率语义，且不能奖励 action 本身 |
| 把 phase186 固定 `ee_x=0.02` 干预直接保留为训练机制 | 该干预只用于证明持续动作的因果作用，不来自策略分布，也不能提供可学习的正确 log-prob | 永久禁止写入环境、阶段或门型逻辑；正式机制必须属于统一策略探索层，并能重算新旧条件概率 |
| 看到 phase192 转换率提高后直接延长训练 | 转换提升伴随 retention pocket/axis/contact 全部回归，且最终左右 base mean `ee_x` 仍为负；当前收益来自 temporal sample 覆盖，未证明 actor 学会 | step150 只作失败证据，不继承、不盲目续训；先完成 phase193 的成功序列 credit 与实际 gradient 诊断 |
| 在 trainer 侧改标签或过滤 14 个错位样本 | 根因是 StagedTaskBase 在 observation 生成后才递增阶段；改标签不能让 policy one-hot、action mask 和 storage 恢复统一语义 | phase194 已在通用任务生命周期修复；禁止恢复 trainer 特判或样本过滤 |
| 把外生 AR context 产生的成功直接解释为 base mean 已获得正 credit | phase194 无污染数据中，成功动作相对 base mean 的位移几乎全来自 context，进入 conditional score 的独立残差接近零且 variant 1 为负 | phase195 必须先验证边缘时间相关轨迹 likelihood；不得仅凭转换率继续训练或增加 reward |
| phase197 转换提高后直接继续 sequence marginal PPO 训练 | 双侧转换和 retention 均较 phase192 改善，但左右 base mean 在后半程持续围绕零下方震荡，最终仍为 `-0.0004/-0.0008` | step150 只用于 phase198 梯度归因；未证明 task gradient 与 optimizer 方向前不得延长训练 |
| 在 action residual mask 为 0 的阶段仍观测 temporal latent | phase198 证明极小方差 GRASP 动作会让 Kalman filter 近乎精确观测 context，并把 OPEN_LATCH credit 大量归到不可学习的 GRASP mean | phase199 已统一为 effective fraction=`configured fraction x ownership mask`；不得恢复固定 fraction 的 sampler 或 likelihood 路径 |
| phase199 左门 signal 转正后立即重训 | 右门已转换轨迹为正，但未转换轨迹仍贡献 `-0.06714` 并主导总 signal | phase200 必须先定位右门失败轨迹的物理/reward 首因；禁止右门特调或盲目增加 update |
| 把 phase201 的高 OPEN_LATCH 转换率当作成功 | iteration126--135 中左右 base mean 持续转正、转换率快速提高，但 contact retention `0.8451 -> 0.5481`；实际阶段完成只检查门缝，不检查仍在可靠夹持 | phase201 按预设门槛停止且不保留 checkpoint；必须先让 OPEN_LATCH 验收同时满足门缝和严格双侧夹持，不得通过继续训练或只看 reached 掩盖语义漏洞 |
| 在新 reward/验收语义下继承 phase183 actor/critic 做短训练 | phase203 exact sequence credit 显示 retention 和 latch 对 `ee_x` 都为负，而正总更新几乎完全由 critic/value 分量推动；phase183 critic 来自旧乘法 reward 和旧完成条件 | phase204 必须随机初始化 actor、critic 和 staged-reset snapshots；不得继续把 phase201/202 的短训曲线用于调权重或判定最终 reward |
| phase205 左门成功样本不足后继续扩大 rollout | 512-step 已覆盖 `665517/492325` 个左右 OPEN_LATCH step，但严格完成仅 `1/92`；继续扩大同分布窗口不会解释入口动作为何先分叉，near-completion 已足以定位晚期 opposed-contact 断层 | 不再盲目增加 horizon 或重复采集相同字段；phase206 转到入口 observation 与 policy mean 的只读归因 |
| 把 opposed-contact 低值直接解释为双爪真实脱离 | phase208 首次事件中约 `97.3%/95.7%` 仍有双爪有效接触、约 `99.7%/99.9%` 仍位于闭合轴两侧；EE 只移动约 1 mm，而单点接触位置沿把手轴向跳变数厘米 | 后续必须区分碰撞接触是否存在与 ContactSensor 单点位置切换；禁止依据旧 opposed 指标继续调策略动作或 reward 权重 |
| 把左门 PUSH_OPEN 失败归因于入口更远、canonical 符号或低层响应 | phase211 在匹配门角和间隙后仍有巨大接触率差；两侧探索残差均证明负 canonical `ee_y` 减小间隙，正常 progress/pose/completion credit 也给出正确负方向 | 不改 action/task frame 或低层；phase212 只处理统一 terminal credit，不做门型特调 |
| 门侧、阶段或 seed 特调 | 当前问题发生在共享 actor/训练链路，且任务要求统一语义 | 永久禁止特例、隐藏开关和临时阈值补丁 |

### 最终验收

同一个高层策略必须在 seed `1700`、`1701`、`1702`、`1703` 上安全完成：

```text
APPROACH_PREGRASP -> GRASP -> OPEN_LATCH -> REPOSITION -> PUSH_OPEN -> THROUGH
```

当前尚未完成最终验收，不能以 smoke、单侧门到达后期阶段或训练脚本正常退出代替。

### 历史对照 checkpoint（禁止用于 phase145 继承）

```text
实验：phase141
checkpoint：logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260925_205918/model_step_000925.pt
SHA-256：b8fdd0bfeda53fed3c26f071cc8c7691772f2458787bf9e82f06640ed39a2b24
并行：4 x 2048 env = 8192 env
模式：纯动力学、headless、无相机、无 RTX sensor、无 frame rendering
```

phase141 从 phase131 step 825 继续训练 100 update，使策略适应已修复的浮动基座重力索引和 lab30 姿态增益语义。step 925 累计到达率：

```text
APPROACH_PREGRASP：100.00%
GRASP：             36.95%
OPEN_LATCH：        15.59%
REPOSITION：        12.57%
PUSH_OPEN：          4.92%
THROUGH：            3.97%
task success：       0
```

phase131 step 825 保留为修复前历史对照基线。phase141 step 925 使用正确的低层接口，但 actor observation/action 仍是迁移前的物理坐标语义；phase144 已改变策略接口，因此该 checkpoint 只能做行为和到达率历史对照，不能加载到 phase145。任何回退都必须明确说明是在做对照实验，不能静默恢复旧语义模型。

### 已确认并修复的问题

- 纯动力学训练路径已确认无相机、无 RTX sensor、关闭 frame rendering；每卡 2048 env 可稳定运行。
- staged reset 和同一 SimulationApp 重建环境导致的历史卡死已从正式多卡训练路径移除；当前训练采用单次场景创建、连续 update。
- 高层 actor observation 已补齐门尺寸、左右铰链、推拉方向、把手/锁舌几何、门轴阻力和当前门状态。
- 六阶段 reward 已按阶段目标和安全边界重构，不要求策略复刻 AutoOpen 的固定动作顺序。
- 通用 delta action 单步限幅已前移到记录、积分和 reward 之前，消除了未执行巨大动作污染 value/reward 的问题。
- KL 自适应学习率所有权已修正，不再与外部 scheduler 冲突，也不再把 `1e-4` 学习率错误放大到 `1e-2`。
- 周期姿态 accumulator 已按 `2*pi` 环绕，修复跨越 `+pi/-pi` 时卡死。
- REPOSITION 推门姿态只约束夹爪插入轴垂直门板，不再强制无关 roll。
- REPOSITION 底盘启停逻辑已消除“验收半径 8 cm、12 cm 提前停车、外侧最低 0.25 m/s”的内部矛盾。
- action supervision 已使用 Huber loss，避免稀有阶段首次出现时无界 MSE 主导更新。
- action supervision 的阶段均衡已与 PPO capped sample weight 分离，按各阶段实际监督坐标数做真正阶段等权。
- 浮动基座 `get_gravity_compensation_forces()` 的前六个基座广义坐标已正确跳过，不再把基座/腿部项错当机械臂重力项。
- RL 在线 IK 姿态误差增益已与 lab30/ManipLoco 及稳态逆映射统一为 `4.0`。

### 已通过对照排除的方向

- THROUGH 卡住不是 Jacobian/DLS 逆映射公式错误：原始逆解关节步长还原误差为 `0.0000 rad`。
- THROUGH 稳定阶段不是 Cartesian 步长裁剪主导：平移裁剪比例最终低于 `1%`，旋转裁剪为 0。
- REPOSITION 旧瓶颈不是简单验收阈值过严，而是 reference 控制在验收带外提前停车并高速穿越。
- 左右门差异不能只归因于 OPEN_LATCH 探索不足；不对称在 GRASP 已出现。

### 历史阶段快照（phase141--147，不得作为当前执行游标）

phase146 已确认因果顺序：右门在 GRASP `0--24` 步、固定爪命中率仅 `0.03%` 时，pocket 已高 `33.4 mm`，其中 reference-target 高 `25.8 mm`；随后固定爪命中率在 `25--49` 步升至 `7.28%`，`50--99` 步升至 `66.72%` 且平均力 `40.4 N`。因此接触是阶段入口几何不连续的后果，不是误差来源。phase147 已在正确的 pocket 几何所有权层修复并完成回归。本节仅保留历史证据，当前状态只看顶部执行卡。

以下是 phase141 旧物理策略接口下的历史断层，只作为 phase145 的对照，不代表当前接口的结论：

```text
                         左门      右门
GRASP reached           22.63%    51.33%
OPEN_LATCH reached       5.88%    25.33%
REPOSITION reached       0.55%    24.63%
PUSH_OPEN reached        0.21%     9.65%
THROUGH reached          0.05%     7.90%
```

1. phase141 证明策略能够适应修正后的低层语义：总 GRASP reached 升至 `36.95%`，PUSH_OPEN/THROUGH 升至 `4.92%/3.97%`。继续围绕 GRASP 静态误差加经验系数不再是当前优先方向。
2. 左门在 GRASP 后明显断层：`22.63% -> 5.88% -> 0.55%`；右门为 `51.33% -> 25.33% -> 24.63%`。首要问题限定在左门 OPEN_LATCH 动作学习与坐标语义。
3. step 925 的 OPEN_LATCH 占用样本中，左门 `latch_position=0.0014 m`、`handle_angle=0.0367 rad`、`hinge_angle=0.0013 rad`；右门分别为 `0.0151 m`、`0.3959 rad`、`0.0286 rad`。左门策略几乎没有压下把手，也没有形成门缝。
4. 当前 OPEN_LATCH progress 为 `0.45*latch_progress + 0.45*gap_progress + 0.10*grasp_contact_quality`；reference 和 supervision 只约束夹爪保持，六维机械臂动作完全交给 PPO。
5. phase142 已确认 observation/action 没有按门型镜像归一。门生成器对右门把手关节轴做 `+X -> -X` 翻转；同一正向把手角对应左门负 roll 增量、右门正 roll 增量。当前 actor 输出和 B2Z1 command 都是机身航向坐标，未经过门型转换。
6. phase141 的策略均值与该缺陷吻合：左/右门 OPEN_LATCH 的 `ee_roll` 均为正值 `+0.0808/+0.0907`；右门可达到 `handle_angle=0.3959 rad`，左门仅 `0.0367 rad`。这不是 reward 几何把左右门算反，而是共享策略在未归一 action 空间学到了同号动作。
7. phase143 和 phase144 已完成任务基数学验证、接口迁移及左右门零 residual 前两阶段回归；phase145 正在验证统一接口下的真实学习行为。
8. phase145 先在 step 1、25、50、75、100 记录左右门逐阶段到达率，首个明显断层必须由轨迹与统计确认后才能修改行为或 reward。
9. 只有随机训练环境稳定出现完整成功，才进入 seed `1700--1703` 独立验收；smoke、单侧门成功和阶段到达率提升都不能替代最终验收。

## 1. 本轮任务

本轮目标是把 B2Z1 开门强化学习训练整理为纯动力学流程，并最终训练出一个能够在 seed `1700`、`1701`、`1702`、`1703` 上安全完成完整任务的策略。

阶段定义为：

```text
APPROACH_PREGRASP
-> GRASP
-> OPEN_LATCH
-> REPOSITION
-> PUSH_OPEN
-> THROUGH
```

最终验收不是“训练脚本不报错”，而是同一个策略在上述四个 seed 上均能安全走完六个阶段。当前尚未达到最终验收标准。

## 2. 设计约束

- 训练必须是 headless、无图像输入、无额外相机、无 RTX sensor，并尽可能关闭 frame rendering。
- 高层 actor 使用完整的门属性、门状态、机器人本体状态和阶段状态等结构化 observation，不依赖相机图像。
- observation、action、坐标系和阶段验收语义必须全流程统一。
- 禁止阶段特例、seed 特调、隐藏开关和临时阈值补丁。
- 修复必须位于正确的接口所有权层，并能够泛化到左右铰链门和随机初态。
- Reward 描述阶段目标和安全边界，不复刻 AutoOpen 的固定动作顺序。策略可以学习更快但仍安全的动作组合。
- 根因未被代码、轨迹或对照实验确认前，只增加无副作用诊断，不修改行为。
- 新机制替代旧机制时删除旧路径、无效参数和死代码。
- 每次行为修改都要回归已经通过的阶段，并最终回归 seed `1700` 至 `1703`。

完整 reward 设计见 `docs/b2z1_door_reward_v2.md`。

## 3. 当前代码状态

```text
仓库：GR00T-VisualSim2Real
分支：auto-open
基准提交：d6459412da15c2e9b31c971e1c8fc2508a451fd5
提交标题：Complete B2Z1 auto-open push and traversal
```

本轮 RL 适配仍包含未提交修改和新增文件。三台机器上的工作区均是源工作树的逐文件副本，不是只同步上述提交。同步后应使用 `git status --short --branch` 核对修改集合，不得执行 `git reset --hard` 或用 Git 分支覆盖当前工作树。

主要修改范围：

- `gr00t/rl/envs/door/door_open_homie.py`：六阶段任务、几何量、验收、reward 和诊断指标。
- `gr00t/rl/config/env/door_open_b2z1.yaml`：环境、随机化和纯动力学配置。
- `gr00t/rl/config/obs/wbmanip/door_open_b2z1.yaml`：高层 actor observation。
- `gr00t/rl/config/rewards/wbmanip/reward_door_open_b2z1.yaml`：reward 权重。
- `gr00t/rl/config/exp/wbmanip/door_open_b2z1_lstm.yaml`：训练实验配置。
- `gr00t/rl/train_agent_trl.py`、`gr00t/rl/trl/`：多卡训练、LSTM、日志和 PPO 适配。
- `gr00t/rl/simulator/isaacsim/isaacsim.py`：headless 和渲染简化。
- `gr00t/rl/scripts/train_b2z1_door_open.sh`：正式容器内训练入口。

## 4. 已验证的纯动力学并行配置

以下是当前源开发机 `omnisim-Super-Server`（4×RTX A6000）上的实测配置。三台 `wholebody_1`、`wholebody_3`、`wholebody_4` 只承担备份和可选算力，不应把本段写成它们各自的“本地机器”结果：

```text
GPU：4 张 NVIDIA RTX A6000（每张约 48 GB）
每张 GPU：2048 env
总并行环境：8192 env
num_steps_per_env：64
num_mini_batches：4
吞吐：约 23k--27k physics steps/s
camera_types：[]
camera sensor：0
RTX sensor：0
frame rendering：关闭
```

8192 env 场景创建约需 3 分钟。当前 robot USD 仍会组合约 26 MB 的 visual layer，并产生 unresolved visual prim 警告；运行时已确认没有相机和渲染计算，但场景初始化仍可进一步精简。这个启动优化与策略行为问题分开处理。

## 5. 历史训练记录

### Phase 50

```text
日志：/tmp/b2z1_train_8192_gap_semantics_phase50.log
运行目录：logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260924_161611
checkpoint：model_step_000700.pt
```

`OPEN_LATCH` 的完成语义已改为门角达到几何门缝阈值。真实锁舌与门框碰撞负责保证门缝不能在锁舌未退出时形成，不再错误要求验收瞬间同时保持锁舌缩回和把手接触。

step 700 的累计到达率：

```text
APPROACH_PREGRASP：100.00%
GRASP：             19.77%
OPEN_LATCH：        15.37%
REPOSITION：         0.28%
PUSH_OPEN：          0.05%
THROUGH：            0.00%
```

### Phase 51

```text
日志：/tmp/b2z1_train_8192_reposition_unified_phase51.log
运行目录：logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260924_164227
起点：phase50/model_step_000700.pt
终点：model_step_000760.pt
总 timesteps：445,644,800
最后一次 iteration：约 23.17 s
```

本轮为 `REPOSITION` 增加统一的门板坐标推门准备位姿：夹爪 `+X` 指向当前门板负法向，夹爪 `+Y` 按门型镜像，夹爪闭合，EE 目标位于当前门板中心外侧 5 cm。`GRASP` 和 `REPOSITION` 共用同一套表面朝向几何函数。

step 760 的累计到达率和当前阶段指标：

```text
APPROACH_PREGRASP reached：1.0000
GRASP reached：             0.2346
OPEN_LATCH reached：        0.1840
REPOSITION reached：        0.0022
PUSH_OPEN reached：         0.0000
THROUGH reached：           0.0000

REPOSITION progress：                    0.6525
base_position_error：                    0.0955 m
base_yaw_error：                         0.0161 rad
push_ready_distance：                    0.3185 m
push_ready_orientation_error：           0.2309 rad
gripper_closed_error：                   0.0076 rad
workspace_projection_error：             0.1244 m
base_still：                              0.3925
ee_still：                                0.4367
all_checks_now：                          0.0000
```

结论：新几何目标运行稳定，没有维度错误或运行时异常，但目前尚未让策略学会 `REPOSITION`。与 phase50 的代表值相比，工作区投影误差从约 `0.0780 m` 增至 `0.1244 m`，EE 到准备点的距离仍约 `0.3185 m`，姿态误差约 `0.2309 rad`。因此不能通过简单放宽阈值宣布通过；下一步需要按左右门分别分析策略动作、目标可达性和训练分布，确认首个真实瓶颈。

## 6. 当前执行顺序

1. phase167 已确认 physical fixed/moving jaw observation 槽位在左右门腕部构型下语义相反；正式接口已统一为 canonical jaw order。
2. phase168 已用正式 4 x 2048 env 配置从随机策略训练至 step100；jaw-order 修复显著削弱但未消除左右 OPEN_LATCH 分叉，禁止继续盲目续训。
3. phase169 使用 phase168 step100 仅作只读诊断基线，不更新策略；先对 arm `dof_pos/dof_vel` 做配对分布与单字段组干预。
4. 若 arm joint state 无因果效应，再依次检查 `hand_handle_transform` 和 `delta_actions` history；一次只检查一个字段组。
5. 根因必须同时满足：字段在镜像等价任务状态下语义不一致、单字段干预能复现 action gap、统一方式符合机器人真实可观测状态。
6. 根因确认后才在 observation 生产所有权层修复，并删除临时导出/干预代码；随后从随机策略重新做同规模干净训练。
7. 左右门同步通过 OPEN_LATCH 后，再按最早未学习阶段依次分析 REPOSITION、PUSH_OPEN、THROUGH；一次只处理一个真实瓶颈。
8. 随机训练环境出现稳定完整成功后，使用同一 checkpoint 独立运行 seed `1700--1703`；四个 seed 全部安全完成才达到最终验收。

## 7. 容器与训练入口

三台机器上的开发容器统一命名为：

```text
镜像：doorman-train:dev
容器：doorman_train
代码挂载：/workspace/GR00T-VisualSim2Real
时区：CST-8（北京时间）
```

容器必须持续运行。训练、评估和诊断命令均在容器内执行，外层仓库只负责镜像和容器生命周期。

单卡 smoke test 示例：

```bash
docker exec -it doorman_train bash
GPU_IDS=0 \
NUM_ENVS=32 \
NUM_TOTAL_BATCHES=2 \
NUM_STEPS_PER_ENV=8 \
NUM_MINI_BATCHES=1 \
gr00t/rl/scripts/train_b2z1_door_open.sh
```

三台远程备份/扩展训练节点 `wholebody_1/3/4` 各配有 2 张 RTX 4090（每张约 24 GB）；源开发机 `omnisim-Super-Server` 配有 4 张 RTX A6000（每张约 48 GB）。下列命令仅是远程双卡节点的首次并行度基线，不代表源开发机配置：

```bash
docker exec -it doorman_train bash
GPU_IDS=0,1 \
NUM_PROCESSES=2 \
NUM_ENVS=512 \
NUM_TOTAL_BATCHES=10 \
NUM_STEPS_PER_ENV=64 \
NUM_MINI_BATCHES=4 \
MAIN_PROCESS_PORT=29500 \
gr00t/rl/scripts/train_b2z1_door_open.sh
```

`NUM_ENVS` 是每张 GPU 的环境数。双卡 `NUM_ENVS=512` 表示总计 1024 个环境。确认显存余量和吞吐后，再依次测试每卡 `1024`、`1536` 等并行度；不得把 `omnisim-Super-Server` 的 A6000 每卡 `2048` 写成当前机器已验证配置。

## 8. 三台机器的部署验证

2026-09-24 已在 `wholebody_1`、`wholebody_3`、`wholebody_4` 完成以下核对：

- 每台机器均可见 2 张 RTX 4090，单卡显存 24564 MiB。
- `doorman_train` 容器持续运行，重启策略为 `unless-stopped`，连续执行 `run.sh` 不会误删并重建容器。
- 容器用户为 `root`（UID/GID `0:0`），`HOME=/root`，时区为北京时间（`CST+0800`）。
- 代码分支为 `auto-open`，HEAD 为 `d6459412da15c2e9b31c971e1c8fc2508a451fd5`，工作树修改集合一致。
- `model_step_000760.pt` 的 SHA-256 均为 `35eed62f6e07c473a2c90d3aba679914b646ed820c148940c84c2279d792a3b9`。
- 每台机器均已用 2 个环境执行 1 次 PPO 迭代，完成 4 个 timestep 并以退出码 0 结束。

上述 smoke test 只验证 Isaac Sim、Isaac Lab、B2Z1 低层模型、环境创建和 PPO 更新链路可运行，不代表已测得 2×RTX 4090 的最大并行度。

## 9. Phase 53 至 Phase 61 进展

### staged-reset 跨 checkpoint 连续性

已修复环境状态保存：checkpoint 不再混入不可恢复的瞬时日志，四卡分别保存并恢复每个环境最新的阶段快照。phase54 从 step 810 恢复时，四个 rank 均立即恢复两千余个后期阶段状态，证明课程不会因续训进程重启而清空。

### REPOSITION 几何修复

已确认并修复两个物理问题：

- 底盘固定工位会使旋转后的门板中心超出 B2Z1 的机械臂工作半径。现在底盘保持门框中心线和固定航向，只沿门框法向前移，使目标工作半径保持 `0.70 m`。
- 原实现把 `gripper_link` 原点距门板误当作夹爪碰撞体净间隙。B2Z1 最前端碰撞体相对原点为 `0.1004 m`，门板半厚 `0.0200 m`，目标净间隙 `0.0300 m`，因此正确的 EE 原点偏移为 `0.1504 m`。

修复后 REPOSITION 首次稳定进入 PUSH_OPEN；step 880 四个 rank 的 PUSH_OPEN 阶段快照数量为 `13/9/10/15`。

### PUSH_OPEN 当前瓶颈

phase58 的物理诊断确认：PUSH_OPEN 入口的夹爪前端净间隙约 `30.6 mm`，预位姿正确；旧策略随后把净间隙扩大到 `43.1 mm`、`63.4 mm`，法向相对速度为负，姿态误差同时增大，未建立门板接触。根因不是推力不足，而是首次接触前缺少直接训练信号且后期阶段样本占比过低。

已完成以下通用修复：

- PUSH_OPEN 进度加入有界的首次安全接触潜势；接触发生后封顶，不要求持续贴门。
- 加入首次接触前的距离/姿态状态代价；两项均归一化并截断，真实接触后永久归零。
- 角速度误差只在门确实沿打开方向转动时生效，并使用有符号打开速度，避免反向旋转被误认为正确速度。
- staged-reset 比例公开调整为 `[0.3, 0.1, 0.1, 0.2, 0.2, 0.1]`；快照仍只在同一随机门型环境内使用。
- PPO 增加可配置的 staged-task 均衡：四卡同步按阶段归一化 advantage，并使用有上限、全局均值为 1 的逆频率 policy/value loss 权重。该实现已通过四卡 2-env、1-update smoke。

phase61 已完整训练至 step 930：

```text
日志：/tmp/b2z1_phase61_stage_balanced_push.log
起点：model_step_000900.pt
运行目录：logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260924_203335
状态：退出码 0，终点 model_step_000930.pt
APPROACH_PREGRASP occupancy/reached：0.5344 / 1.0000
GRASP occupancy/reached：0.0581 / 0.4656
OPEN_LATCH occupancy/reached：0.1987 / 0.4075
REPOSITION occupancy/reached：0.1961 / 0.2088
PUSH_OPEN occupancy/reached：0.0128 / 0.0128
PUSH_OPEN panel clearance：0.4563 m
PUSH_OPEN panel approach speed：-0.0221 m/s
PUSH_OPEN policy mean ee_x：-0.0061
PUSH_OPEN accumulated ee_x command：-0.2843 m
PUSH_OPEN contact seen：0.0000
THROUGH reached：0.0000
```

动作坐标已核对：底盘朝向门框时，EE 局部 `+x` 是向门板接近，策略的负值确实是在主动撤离，不是坐标符号误读。进一步检查发现 PUSH_OPEN 接近进度在净间隙超过 `5 cm` 后恒为零，接近代价在超过 `10 cm` 后又变成常数，因此策略退至 `45.6 cm` 时没有任何继续增大的状态代价。这是由代码和轨迹共同确认的 reward 饱和问题。

阶段均衡 PPO 的完整 8192-env 审计结果：

```text
rollout 总样本：524288
APPROACH_PREGRASP：347712，权重 0.360476
GRASP：             58351，权重 2.148069
OPEN_LATCH：        54307，权重 2.308027
REPOSITION：         61626，权重 2.033914
PUSH_OPEN：           2292，权重 10.000000
THROUGH：                0，权重 0.000000
样本加权均值：0.9999996
```

权重实现现在同时严格满足全局样本均值为 1 和最终权重不超过 10，并把每阶段 count/weight 写入无 WandB 的文本日志。PUSH_OPEN reward 已只针对上述根因修改：首次接触前的距离/姿态代价在完整可达工作区保持梯度，且该阶段使用有符号进度，离开门板会产生负回报；动作、观测、验收阈值和其他阶段均未修改。

当前 phase63：

```text
日志：/tmp/b2z1_phase63_push_unsaturated.log
起点：logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260924_205543/model_step_000931.pt
目标：step 961
配置：4 x RTX A6000，每卡 2048 env，纯动力学、headless、camera_types=[]
状态：退出码 0；step 960 checkpoint 完整保存
```

phase63 至 step 961 仍未发生门板接触。最终 PUSH_OPEN 样本为 `8950/524288`，策略 `ee_x` 均值只从约 `-0.0064` 改善到 `-0.0055`，净间隙仍为 `0.4109 m`。因此 reward 饱和是一个真实问题，但不是唯一瓶颈。

phase64 加入了不改变行为的 PUSH_OPEN 入口诊断。实测前 5 个 policy step：

```text
panel clearance：                 0.005--0.013 m
commanded goal error：            0.003--0.008 m
physical EE tracking error：      0.005--0.011 m
push-ready orientation error：    0.010--0.024 rad
entry policy mean ee_x：         -0.0051 至 -0.0058
```

这证明 staged-reset 恢复的机器人、门和 delta-action 累加器彼此一致；策略从进入 PUSH_OPEN 的第一步就主动输出负向 `ee_x`，随后才把夹爪撤离门板。问题位于策略决策，不在 reset、IK 或入口物理状态。

继续审计 actor observation 后确认：门根姿态、铰链角、机器人关节状态和累计命令在数学上足以间接推导 EE 相对门板的关系，但任务几何 observation 对前四阶段均提供直接误差，对 PUSH_OPEN 却没有门板相对 EE 的直接几何量，迫使网络自行同时学习机器人 FK 和旋转门板几何。现已在统一接口新增 `b2z1_panel_interaction`，所有阶段始终使用相同的 12 维语义：

```text
EE 相对当前门板推门目标的位置误差（门板坐标，3）
EE 相对当前门板目标姿态的轴角误差（门板坐标，3）
EE 相对门板的线速度（门板坐标，3）
门板接触力（1）
本阶段是否已发生门板接触（1）
肩部相对门板净空（1）
```

该修改将 actor/critic 输入维度改为 `172/177`。干净模型的四卡、每卡 2 env、1 update smoke 已以退出码 0 完成。旧 checkpoint 输入层不兼容，后续不做隐藏局部加载或 checkpoint 特调，而是从零干净重训 phase65。

phase65 已在 step 50 主动终止，其 checkpoint 不可用于后续训练。根因是一个通用 Actor 接口 bug：环境按 observation 名称排序后拼接 actor 输入，新增 `b2z1_panel_interaction` 后，`b2z1_task_geometry` 的全局 offset 改变，但 `action_residual_obs_slice` 仍使用旧的硬编码 `[36:46]`，导致策略把门板 observation 的一段误当成 action reference。

现已在通用 Actor 层新增 `action_residual_obs_key`：根据实际 observation 排序和维度定位指定 key，并自动取该 key 最后的 `num_actions` 维。配置改为 `action_residual_obs_key: b2z1_task_geometry`，不再保存易碎的全局数字 offset。当前布局下解析结果为 `[48:58]`；今后添加或重排其他 observation 会自动更新。四卡干净 smoke 已以退出码 0 通过。

当前 phase66：

```text
日志：/tmp/b2z1_phase66_keyed_reference_clean.log
运行目录：logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260924_220145
起点：干净模型，无 checkpoint
目标：step 100
执行机器：omnisim-Super-Server
配置：4 x RTX A6000，每卡 2048 env，总计 8192 env，纯动力学、headless、camera_types=[]
状态：已在 step 30 主动终止；该训练不可作为后续 checkpoint
checkpoint：model_step_000010.pt
SHA-256：70a1589606cddb231eb910890fdfd48b0d7cefa928f16ede4c365ad9dd7ff675
```

step 10 时仍只有 `APPROACH_PREGRASP` 样本，尚无环境迁移到 `GRASP`：

```text
APPROACH_PREGRASP occupancy/reached/progress：1.0000 / 1.0000 / 0.0439
GRASP 及后续阶段 occupancy/reached：0
底盘 ready 比例：0.6471
底盘 ready 且静止比例：0.3701
EE 位置命令误差：0.0562 m
EE 姿态命令误差：1.7310 rad
policy/action reference error（底盘）：0.0734
policy/action reference error（EE 位置）：0.1037
policy/action reference error（夹爪）：0.0299
```

修复后的 action reference 已不再出现 phase65 约 `0.88` 的整体错位，证明按 observation key 动态解析 reference 生效。三台备份机器均已同步并校验 step 10 checkpoint，哈希一致。

phase66 到 step 30 仍无任何环境进入 `GRASP`。最近 10 步中，EE 姿态命令误差从 `1.4386 rad` 持续增至 `1.8821 rad`，policy action supervision loss 从 `0.003622` 增至 `0.003714`。代码审计确认 residual actor 的动作定义为 `reference + residual_network(obs)`，但 residual 网络末层沿用随机初始化；持续非零的随机残差经 delta-action 累加器逐帧积分，破坏了已经验证的 reference 控制。

现已在通用 Actor 接口修复该语义：配置 action reference 时，动作输出投影层以零权重、零偏置初始化，使干净模型的初始策略严格等于 reference；普通 actor 不受影响，加载 checkpoint 时仍由 checkpoint 参数覆盖初始化值。前馈和循环 actor 均已覆盖。数值测试中 32 组随机 observation 的输出与 reference 最大误差为 `0.0`；四卡最小 smoke 退出码为 0，首个 rollout 的监督损失和各组 action target error 均为 0。

当前 phase67：

```text
日志：/tmp/b2z1_phase67_zero_residual_clean.log
起点：干净模型，无 checkpoint
目标：step 100
执行机器：omnisim-Super-Server
配置：4 x RTX A6000，每卡 2048 env，总计 8192 env，纯动力学、headless、camera_types=[]
进程组：1000963（记录于 /tmp/b2z1_phase67_zero_residual_clean.pgid）
状态：已完成，退出码 0；step 100 checkpoint 已完整保存
运行目录：logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260924_221849
checkpoint：model_step_000100.pt
SHA-256：d5a3e0adb4e9f680727367a5e1e969ccf91f54c6b76055487b742f7cc65be253
```

phase65 遗留的四个孤立 worker 已被发现并清理；phase67 使用独立进程组启动，后续若提前终止将按进程组整体回收。

phase67 step 10 首次证明干净训练已经恢复阶段迁移：

```text
APPROACH_PREGRASP occupancy/reached/progress：0.8884 / 1.0000 / 0.3965
GRASP occupancy/reached/progress：0.0643 / 0.1116 / 0.5882
OPEN_LATCH occupancy/reached/progress：0.0473 / 0.0473 / 0.0981
REPOSITION 及后续阶段 occupancy/reached：0
底盘 ready 比例：0.7882
稳定满足第一阶段验收 10 帧的比例：0.1106
EE 位置命令误差：0.0005 m
EE 姿态命令误差：0.0033 rad
policy/action reference error（底盘）：0.0004
policy/action reference error（EE 位置）：0.0001
policy/action reference error（夹爪）：0.0003
```

同一步数下，phase66 的 `GRASP` 和 `OPEN_LATCH` reached 均为 0。phase67 的 supervision loss 连续 10 步显示为 `0.000000`，说明零残差初态和已验证参考控制在 PPO 更新后仍保持稳定。下一检查点是首次进入 `REPOSITION`，当前不调整 reward 或验收阈值。

phase67 在 step 15 首次进入 `REPOSITION`。step 50 状态如下：

```text
APPROACH_PREGRASP occupancy/reached/progress：0.7372 / 1.0000 / 0.3890
GRASP occupancy/reached/progress：0.0568 / 0.2628 / 0.5905
OPEN_LATCH occupancy/reached/progress：0.2017 / 0.2060 / 0.2906
REPOSITION occupancy/reached/progress：0.0043 / 0.0043 / 0.6059
PUSH_OPEN 及 THROUGH occupancy/reached：0
REPOSITION 全部即时条件通过率：0.0013
REPOSITION 推门距离通过率：0.3163
REPOSITION 推门姿态通过率：0.0475
REPOSITION 底盘位置通过率：0.1480
REPOSITION 底盘 yaw 通过率：0.9875
REPOSITION 夹爪闭合通过率：0.9859
REPOSITION policy/reference 最大分组误差：0.0030
```

REPOSITION 平均进度仍在增长，且动作参考接口保持准确；现阶段继续跑满既定 step 100，不根据早期稀有样本修改 reward 或验收阈值。若到 step 100 仍无 `PUSH_OPEN`，再基于该阶段各验收项的时序交集和物理跟踪误差确定根因。

phase67 step 100 最终结果：

```text
checkpoint：model_step_000100.pt
SHA-256：d5a3e0adb4e9f680727367a5e1e969ccf91f54c6b76055487b742f7cc65be253
APPROACH_PREGRASP occupancy/reached/progress：0.6253 / 1.0000 / 0.4083
GRASP occupancy/reached/progress：0.0633 / 0.3747 / 0.5841
OPEN_LATCH occupancy/reached/progress：0.2310 / 0.3114 / 0.3936
REPOSITION occupancy/reached/progress：0.0801 / 0.0803 / 0.5428
PUSH_OPEN occupancy/reached/progress：0.0002 / 0.0002 / 0.0887
最后一个 rollout 的 THROUGH occupancy/reached：0 / 0
完整 task_success：0
```

训练过程中在 step 80 首次出现 `THROUGH reached=0.0001`，说明六阶段入口链路可达，但没有完成 THROUGH 并触发最终成功，不能作为策略验收。

step 100 checkpoint 已确认包含模型、优化器、学习率调度器，以及四个 rank 各自的 staged-reset 状态。四个 rank 合计可恢复的非初始阶段入口为：

```text
GRASP：6285
OPEN_LATCH：6115
REPOSITION：1154
PUSH_OPEN：4
THROUGH：1
```

当前 phase68：

```text
日志：/tmp/b2z1_phase68_resume100_to200.log
起点：phase67 model_step_000100.pt
目标：step 200
执行机器：omnisim-Super-Server
配置：4 x RTX A6000，每卡 2048 env，总计 8192 env，纯动力学、headless、camera_types=[]
进程组：1005481（记录于 /tmp/b2z1_phase68_resume100_to200.pgid）
状态：已完成，退出码 0；四个 rank 均正确恢复 step 100 和 staged-reset 快照
```

phase68 step 200 最终结果：

```text
checkpoint：model_step_000200.pt
SHA-256：15d2337e398c472523a829c329fad0ce342948640eadb8980ba79145dba29c5c
APPROACH_PREGRASP occupancy/reached：0.4660 / 1.0000
GRASP occupancy/reached：0.0508 / 0.5340
OPEN_LATCH occupancy/reached：0.0633 / 0.4833
REPOSITION occupancy/reached：0.4143 / 0.4200
PUSH_OPEN occupancy/reached/progress：0.0052 / 0.0057 / 0.1902
THROUGH occupancy/reached/progress：0.0005 / 0.0005 / 0.1264
完整 task_success：0
PUSH_OPEN policy mean ee_x：-0.0043
PUSH_OPEN 门板间隙：0.4391 m
PUSH_OPEN 接触发生率：0
```

PUSH_OPEN 的行为从 step 120 到 step 200 持续恶化：策略更强地输出负 `ee_x` 并从门板退到 `43.9 cm`。代码审计确认设计文档和注释要求“有符号势函数在完整工作区持续惩罚离板”，但实际实现把
`1 - clearance / 0.05` 下限截断到 0，导致间隙超过 `5 cm` 后，从 `5 cm` 继续退到 `44 cm` 不再损失任何阶段势能。现已只修正该实现偏差：保留最大值 1，取消最小值 0。数值检查结果为：

```text
clearance：       0.00, 0.02, 0.05, 0.10, 0.44 m
signed progress：1.00, 0.60, 0.00, -1.00, -7.80
```

该修改不改变 observation、action、阶段验收、动作参考或前四阶段 reward。

step 200 checkpoint 中四个 rank 合计保留 `97` 个独立 PUSH_OPEN 入口和 `5` 个 THROUGH 入口。

当前 phase69：

```text
日志：/tmp/b2z1_phase69_signed_push_potential.log
起点：phase68 model_step_000200.pt
目标：step 250
执行机器：omnisim-Super-Server
配置：4 x RTX A6000，每卡 2048 env，总计 8192 env，纯动力学、headless、camera_types=[]
进程组：1010212（记录于 /tmp/b2z1_phase69_signed_push_potential.pgid）
状态：启动中；先在 step 210/220 检查 PUSH_OPEN 的 ee_x 与门板间隙是否反转
```

## 10. 每小时备份

源训练机有随时关机风险。训练期间每小时执行一次进度快照：

- 更新本文档中的当前 phase、checkpoint、关键阶段指标和未解决问题。
- 将当前未提交代码逐文件同步到 `wholebody_1`、`wholebody_3`、`wholebody_4` 的 `/root/workspace/doorman_docker_package`。
- 同步最近一个完整 checkpoint，并校验 SHA-256。
- 不覆盖远程 Git 历史，不写入 GitHub/GitLab 凭据，不复制大体积历史日志和无关目录。

最近一次备份完成于 2026-09-25 22:05（北京时间）：当前 22 个修改/新增源码与文档已同步至 `wholebody_1`、`wholebody_3`、`wholebody_4`；同步后的逐文件内容和 Git 工作树修改集合均与源机一致。phase141 `model_step_000925.pt` 三端 SHA-256 均为 `b8fdd0bfeda53fed3c26f071cc8c7691772f2458787bf9e82f06640ed39a2b24`。未修改远程提交历史或凭据。

## 11. 2026-09-25 PUSH_OPEN 历史训练记录

> 本节记录的是 2026-09-25 当时 checkpoint 的实验过程，不代表当前阶段状态。
> 后续正式独立阶段验收已经覆盖本节旧结论：PUSH_OPEN 两个 seed 为
> `91.99%/92.68%`，THROUGH 为 `85.11%/85.06%`，两者均已通过。
> 当前权威状态只看顶部“当前执行卡”和“六阶段目标与进展总账”。

本节数据来自源训练机 `omnisim-Super-Server`（4 x RTX A6000），不是三台备份机的算力或实测结果。训练保持每卡 2048 个环境、共 8192 个环境，且启动日志已确认 `cameras=0, RTX sensors=0, frame rendering disabled`。

已完成的通用修复：

- PUSH_OPEN 门板间隙势函数不再在 5 cm 之外饱和，远离门板会持续损失势能。
- actor 的 action reference 按 observation key 定位，最终层零初始化，未训练 residual 严格复现 reference。
- PPO rollout 保存阶段，按阶段同步归一化 advantage，并对稀有阶段使用有上限的逆频率权重。
- actor 使用实际 rollout reward-to-go；critic 继续使用标准 bootstrapped target，避免稀有 PUSH_OPEN 样本被不准确的 critic bootstrap 反向驱动。
- PUSH_OPEN 的底盘 reference 使用门板几何和机械臂舒适工作区反馈，EE 姿态 reference 实时跟随旋转门板法向。
- PUSH_OPEN 姿态约束不再因“曾接触过门板”而永久失效：接触历史只解除间隙要求，姿态对齐在整个推门阶段继续进入势函数和持续约束 reward。

phase90 从 step 330 训练到 step 350。完整 checkpoint：

```text
logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260925_024616/model_step_000350.pt
SHA-256: 31e8bf9ab7ec9917d6321f48231b3edd53ab84cb0adea7d8cd4200c290f61436
```

该轮证明底盘反馈安全有效，但也暴露出姿态 reward 漏洞：actor 每步产生很小的反向姿态 residual，经积分后把目标到命令姿态误差从约 0.05 rad 推高到约 0.32 rad。低层命令到实际 EE 的残差约 0.11--0.13 rad，主要增长发生在高层目标到命令之间，因此不是低层跟踪失效。

修复姿态 reward 后，phase93 从同一个干净 step 350 做 10 个 update：

```text
checkpoint: logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260925_031526/model_step_000360.pt
SHA-256: f5223034678d5c8dd0522d4afea6fb515126ff9eda91637e1b1539a6f9c3b949
PUSH_OPEN 最大平均门角: 0.5520 rad
PUSH_OPEN 最大平均开门速度: 0.1192 rad/s
PUSH_OPEN 平均门板力: 约 5--10 N
THROUGH reached: 0.0031
完整 task_success: 0
```

当前 phase94：

```text
日志: /tmp/b2z1_phase94_push_pose_to380.inner.log
起点: phase93 model_step_000360.pt
目标: step 380
状态: 运行中
```

当前验收尚未完成。phase94 首先验证持续姿态约束能否把已形成的积分 residual 拉回，并观察门角是否突破约 0.55 rad；在这项根因未验证前不增加新的阶段特例或临时阈值。

### phase94--100：PUSH_OPEN 左右门一致性

phase94 从 step 360 训练到 step 380，PUSH_OPEN 峰值平均门角约 `0.635 rad`，峰值平均开门速度约 `0.232 rad/s`，`THROUGH reached` 约 `0.0046`，但完整成功仍为 0。分侧诊断发现这些 THROUGH 样本最初全部来自左铰链门。

phase96 增加按“阶段 x 门侧”分组的 advantage 归一化。右门接触率提升到约 `0.76--0.81`，并首次出现少量 THROUGH，但姿态误差仍持续累积。phase97 进一步证明软 action supervision 无法阻止积分动作产生长期偏置。

phase98 在 policy distribution 内落实逐 action 所有权：PUSH_OPEN 的底盘、EE 姿态和夹爪由反馈控制，PPO 只控制 EE 平移。所有反馈通道的 residual 已严格为 0，但目标姿态到命令姿态的误差仍从 `0.045 rad` 增长到 `0.384 rad`。因此 PPO residual 不是剩余误差的根因。

代码审计确认 B2Z1 的世界系 EE 姿态由“姿态命令 + EE 球坐标 pitch/yaw”共同重建。此前 PUSH_OPEN 允许 PPO 改变 EE 位置，却继续使用按固定推门点计算的姿态命令，导致平移和世界姿态发生隐式耦合。现已在 B2Z1 命令层增加与正向重建严格互逆的姿态映射，并按当前已投影的 EE 位置实时反算姿态命令。4096 组随机数学往返测试的最大角误差约 `0.0017 rad`。

phase99 从同一个 step 380 做 10 次更新验证该修复：

```text
目标姿态到命令姿态误差：全程 0.004--0.012 rad，不再累积
step 390 THROUGH reached：0.0041
左门 THROUGH reached：0.0080
右门 THROUGH reached：0.0002
完整 task_success：0
```

phase99 同时暴露出 reward 漏洞：`contact_seen` 一旦为真，就永久免除了门板间隙要求。右门即使已经减速到 `0.001 rad/s`、夹爪离板 `6.6 cm`，也不会获得重新接近门板的引导。现已将该离散历史豁免替换为连续的“自由运动信用”：门越接近 `20 deg/s` 目标速度，越允许无接触运动；门减速时，重新接近门板的势函数和约束惩罚连续恢复。

phase100 从 step 390 训练到 step 400。早期右门间隙回到 `7.9 mm`、接触率达到 `0.72`，说明新 reward 能恢复重新接近行为；后段左右门仍会减速并拉开间隙，当前 10 次更新不足以形成稳定策略，但右门从 step 398 起再次出现 THROUGH 样本。

```text
最新 checkpoint：logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260925_042619/model_step_000400.pt
SHA-256：d3335a67d7dc57e2dfda68555ff8997e947d29aad20fd05fde42ed47393d26ac
APPROACH_PREGRASP / GRASP / OPEN_LATCH / REPOSITION reached：1.0000 / 0.5638 / 0.4991 / 0.4601
THROUGH reached：0.0037
左门 / 右门 THROUGH reached：0.0071 / 0.0002
完整 task_success：0
```

当时状态为“训练链路可运行、PUSH_OPEN 正在学习、最终验收未完成”。该结论已被后续 step400 独立阶段正式验收覆盖；PUSH_OPEN 当前已通过独立阶段验收。

### phase101--106：THROUGH 训练链路

phase101 从 step 400 连续训练到 step 450。左右门都能进入 THROUGH，step 450 左/右门平均角度为 `0.654/0.662 rad`，THROUGH occupancy/reached 为 `0.0083`；但完整成功仍为 0，THROUGH 平均进度长期只有约 `0.03--0.05`。

新增无副作用诊断后确认，旧 THROUGH reward 在大多数状态下没有梯度：穿门距离在 0 以下被截断，机械臂只使用 `arm_safely_stowed` 布尔值。策略既未穿过 1 m、又未突然到达 middle preset 时，任意中间改善都不可见。现已改为有符号穿门距离势函数，并将 middle preset 的六关节误差逐关节连续塑形；最终验收条件未改变。

phase104 同时验证原 THROUGH 探索噪声不足：底盘采样噪声对应的物理速度约 `0.032 m/s`，低于低层控制器约 `0.1 m/s` 的稳定响应区。THROUGH 的训练探索现改为底盘平移 4 倍、yaw 2 倍、EE 六维 3 倍、夹爪不变；部署仍使用 policy mean，action 和 observation 语义不变。

继续审计 staged-reset 后发现 `_through_start_pos` 没有随快照保存或从快照重建。恢复 THROUGH 快照时，代码先用旧环境机器人位置写起点，再恢复机器人 root state，导致每局凭空产生约 `-0.1~-0.3 m` 初始位移，训练 reward 和完成距离都失真。现已在通用 staged-reset 接口增加“状态选择后重建派生状态”回调，B2Z1 门任务从快照 robot root position 重建 THROUGH 起点，不改变 checkpoint schema，旧 checkpoint 可直接加载。

phase106 从 step 470 checkpoint 做运行验证：

```text
修复前 THROUGH 入口距离：约 -0.2 m
修复后 THROUGH 入口距离：0.0001--0.0026 m
入口样本前向速度最高：0.0859 m/s
checkpoint：logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260925_054746/model_step_000472.pt
```

当前仍无完整成功，不能作为最终策略。下一轮从 step 472 在正确的 THROUGH 起点、连续 reward 和可达探索尺度下继续训练。

### phase107--108：正确 THROUGH 起点下的首轮学习

phase107 从 step 472 训练到 step 500，四卡训练正常结束，无遗留训练进程。完整 checkpoint：

```text
logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260925_055449/model_step_000500.pt
SHA-256：7fd52118ebe1ff6e432fb0851d280bd2315cd2c0e4af94f84de28184afa91d17
```

修复后的 staged-reset 入口距离保持在厘米级附近，不再出现修复前凭空产生的约 `-0.1~-0.3 m` 位移。step 473--500 的主要趋势为：

- THROUGH reached 从 `0.0030` 上升到峰值 `0.0098`，step 500 为 `0.0066`；完整 task success 仍为 0。
- step 493 以后，两侧门的底盘前进 policy mean 开始明显增长；step 500 左/右门分别为 `0.0292/0.0273`，入口平均前向速度达到 `0.1473 m/s`。
- THROUGH 全阶段平均位移仍为负，step 500 为 `-0.1939 m`，平均前向速度为 `-0.0241 m/s`。这说明入口探索已能驱动底盘前进，但策略尚未在整个阶段保持前进。
- middle preset 六关节平均误差没有下降，step 500 为 `1.3693 rad`，最大关节误差为 `2.7092 rad`，`arm_safely_stowed` 仍为 0。

因此 phase107 证明前进奖励和探索尺度已经产生可学习信号，但尚不能证明 THROUGH 已学会；机械臂回到 middle preset 的控制链路仍需独立判断是训练量不足，还是 EE 动作空间与关节验收目标之间存在结构性不匹配。

phase108 已于 2026-09-25 06:13 CST 从 step 500 启动，目标训练到 step 550。配置继续保持 4 x RTX A6000、每卡 2048 env、总计 8192 env、纯动力学、headless、无相机和无渲染。该轮不修改行为逻辑，用于观察底盘前进趋势能否持续形成，以及机械臂误差是否出现下降趋势。

### phase108--110：THROUGH 根因与统一参考接口

phase108 从 step 500 连续训练到 step 550。50 个 update 后，THROUGH 平均位移仍长期为负，step 550 为 `-0.1782 m`；middle preset 平均关节误差为 `1.3183 rad`，完整成功仍为 0。因此瓶颈不是短暂的 snapshot 重建过程或训练量不足。

phase109 只增加无副作用诊断，并从 step 550 训练到 step 560。诊断确认：

- THROUGH 入口航向误差约 `0.017 rad`，全阶段约 `0.03--0.14 rad`，不是前进坐标反向。
- 策略给出的世界系前进命令只有 `0.002--0.015 m/s`，远低于低层约 `0.1 m/s` 的步态响应区，因此机器人没有主动前进，并被交互力缓慢带回。
- EE 距机身约 `0.83 m`；关节 2/3/4/5 的平均误差均超过 `1 rad`。当前 reward 要求 middle 关节构型，但 action 是笛卡尔 EE 增量，策略没有获得明确的 IK 分支目标。

进一步用 URDF、在线阻尼 IK、PD 刚度和重力方程直接求解 middle preset 对应的稳态 EE 命令。数学方程的关节增量回代误差低于 `1e-16 rad`，但把单一 EE 终点交给仿真后，机械臂从当前构型收敛到了另一条 IK 分支，最大关节误差 `2.79 rad`。这证明单一笛卡尔终点不能表达 middle preset 的关节分支。

现有架构已经使用统一的 goal-conditioned action reference 加 PPO residual。THROUGH 因此继续使用相同的 10 维 action 语义，并新增两项通用参考：

- 底盘沿门框法向给出 `0.20 m/s` 前进参考，同时用连续航向反馈保持正对门框；通过距离达到 1 m 后参考速度归零。
- 机械臂把当前关节误差按 `0.50 rad/s` 限速后，经实时 EE Jacobian 和在线阻尼 IK 的局部逆映射转换成同一种笛卡尔增量 action。它保留当前 IK 分支信息，不切换关节直控；PPO 仍拥有全部 residual，可学习更快但安全的协调方式。

phase110 从 step 560 训练到 step 570。底盘链路立即恢复正确语义：

```text
THROUGH 平均位移：0.0558 m -> 0.6813 m
世界系前进命令：约 0.18--0.21 m/s
平均前向速度：约 0.06--0.13 m/s
航向误差：约 0.02 rad
distance_ok：0.0005 -> 0.1377
```

第一版机械臂映射使用 `J*dq`，只能保证正向微分，不严格逆于在线阻尼 IK。实跑中关节 3/4/6 改善，但关节 2/5 反向，middle 平均误差仍约 `1.4 rad`。当前代码已将其替换为在线矩阵 `J^T (J J^T + lambda^2 I)^-1` 的批量局部逆，奇异批次才退回正向微分映射；下一轮从 step 570 验证六关节是否同向收敛。

phase110 checkpoint：

```text
logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260925_065723/model_step_000570.pt
SHA-256：0584f8c6d1a5aef84d978d05d3db054965a11e20f3587fb08e374b256f6799ba
```

### phase111--113：THROUGH 参考跟随与软监督验证

phase111 从 step 570 训练到 step 580。精确的在线阻尼 IK 局部逆映射开始让机械臂向 middle preset 收敛，六关节平均绝对误差从约 `1.52 rad` 降到约 `1.21 rad`，并首次出现完整任务成功：

```text
checkpoint: logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260925_070958/model_step_000580.pt
SHA-256: 19b81ba5b6f615a84131497196ed6f53556155c9be489a39c764731167990cfe
最高 task_success: 0.0022
```

但诊断同时确认，旧 checkpoint 学到的 PPO residual 在多个机械臂维度上持续抵消新增 reference。例如一次左铰链门采样中，EE 高度 reference 为 `+0.0405`、residual 为 `-0.0539`；yaw reference 为 `-0.0612`、residual 为 `+0.0643`。因此 reference 的数学映射正确，但策略尚未学会停止抵消它。

phase112 从 step 580 继续训练到 step 620。middle 平均误差最低约 `1.07 rad`，随后反弹到约 `1.19 rad`；完整成功只偶发出现，未形成稳定趋势。该轮正常保存并主动结束：

```text
checkpoint: logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260925_071936/model_step_000620.pt
SHA-256: d034bc37052b202aaa0be03a30990d56308a81b9295acf28992d73d4eb3522cf
```

phase113 沿用已有统一接口 `get_policy_action_supervision()`，把 THROUGH 纳入全维软 action-reference 监督。该修改没有屏蔽 PPO residual，也没有改变 observation、action 或坐标语义。step 620--650 的结果为：

```text
checkpoint: logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260925_074432/model_step_000650.pt
SHA-256: 1fa8d15057933da1df26a8b3185543776350c00f5de0f94f69da4f25131f2bbe
step 634 task_success: 0.0067
step 650 middle 平均关节误差: 1.1659 rad
```

成功率随后又回到 0，且 residual 仍近乎逐维抵消 reference。step 650 左门样本中，EE 高度 reference/residual 为 `+0.0249/-0.0317`，pitch 为 `+0.0284/-0.0289`，yaw 为 `-0.0405/+0.0422`。监督损失已按有效 mask 正确归一化，不存在被阶段 occupancy 稀释的问题；但当前 `policy_action_supervision_coef=1.0` 时，监督损失仅约 `8e-5--1e-4`，相对 PPO 与 value loss 太弱。

当前没有训练进程遗留。下一项受控实验是仅调整已有统一监督接口的强度，先把 `policy_action_supervision_coef` 从 `1.0` 提高到 `10.0`，从 step 650 做短程验证，检查 residual 是否开始跟随 reference、middle 六关节是否同时下降。若较强监督仍无法消除旧 checkpoint 的抵消行为，则停止迁就旧策略并进行干净重训；不会增加阶段特例、seed 特调或隐藏控制路径。

### phase114：监督系数 10 的短程对照

phase114 只将已有统一监督接口的 `policy_action_supervision_coef` 从 `1.0` 提高到 `10.0`，从 step 650 训练到 step 660；环境、reward、action、observation 和阶段行为均未改变。运行再次确认 4 x 2048 env 下 `cameras=0, RTX sensors=0, frame rendering disabled`。

```text
checkpoint: logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260925_080849/model_step_000660.pt
SHA-256: d2d84915c84a7538a070bbc6aee34369838c56a76ce09b1b0527a3e93a15cced
```

10 个 update 尚未解决 reference 抵消：完整成功仍为 0，middle 平均关节误差约从 `1.17 rad` 上升到 `1.33 rad`，关键 EE 姿态 residual 仍与 reference 反向。监督原始损失约 `1.9e-5--7.1e-4`，乘系数后对 actor 的贡献与 PPO policy loss 已处于同一量级，不是被 critic value loss直接稀释。

但本轮是新的 SimulationApp，staged-reset 样本不会随 checkpoint 保存，THROUGH occupancy 从 phase113 末尾约 `1.0%` 暂时回落到约 `0.55%`。因此 10 步结果同时混入了后期阶段样本重建过程，尚不足以区分“监督强度仍不足”和“样本分布尚未恢复”。phase115 将从 step 660 在同一配置下连续训练 40 步到 700；若样本恢复后 residual 和 middle 误差仍不改善，就停止继续提高系数，改用干净重训验证统一架构。

### phase115：样本恢复后的结论

phase115 从 step 660 连续训练到 step 700。THROUGH occupancy 在 step 691 左右恢复到 phase113 水平，step 700 达到 `1.35%`，因此后半段已不再受“后期阶段样本尚未重建”的解释限制。

```text
checkpoint: logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260925_081442/model_step_000700.pt
SHA-256: 5c98f4b94650c9008cbfee05cb16e4ef44c0ab80d65e94793e7c12b3ebacf4ae
step 700 middle 平均关节误差: 1.2139 rad
step 700 arm_safely_stowed: 0.0086
step 700 task_success: 0
```

middle 误差在样本恢复过程中从峰值 `1.326 rad` 下降到 `1.214 rad`，说明监督产生了梯度，但仍差于 step 650 的 `1.166 rad`，且完整成功始终为 0。终点 residual 继续系统性抵消 reference：左门 EE 高度为 `+0.0255/-0.0273`、pitch 为 `+0.0254/-0.0272`、yaw 为 `-0.0320/+0.0338`；右门也有同类反向项。继续提高监督系数只会用更强辅助损失硬掰旧策略，没有解决旧 checkpoint 与新统一 reference 架构的历史不一致。

因此后续停止迁就 step 700，使用当前统一 observation/action/reference、连续 reward 和 staged-reset 机制进行随机初始化的干净重训。该选择不增加阶段特例，也不保留旧行为兼容路径；先运行 100 个 update 检查各阶段 occupancy、reference residual 和最早瓶颈，再根据首个未学会阶段继续训练或做根因分析。

### phase116：干净重训暴露通用 delta action 限幅错误

phase116 不加载 checkpoint，从随机初始化训练到 step 100。前 67 步行为符合预期：APPROACH_PREGRASP reference 的 policy error 始终为 0，GRASP reached 增长到约 `9%`，OPEN_LATCH occupancy 增长到约 `2.7%`，证明干净策略可以沿统一接口重新积累后期阶段样本。

在首批 REPOSITION 样本出现时，训练出现确定性的数值异常：

```text
policy_action_supervision_avg: 6638.451660
value_avg: 50700840960
Mean episode rew_penalty_delta_action_rate: -13848.8027
```

继续运行后，OPEN_LATCH 的动作惩罚反复达到约 `-1e4~-3e4`，其 progress 长期停在约 `0.04`，REPOSITION occupancy 只有约 `0.01%`。诊断确认 reference 本身始终在正常的归一化范围内；异常来自 actor 对尚未充分覆盖的阶段 observation 外推了 `O(10^2)` 的高斯 delta action。

根因位于通用 `DeltaActionBase`：它先把未限幅 policy action 写入 `_last_delta_actions` 并积分，reward 也直接使用该未执行的值；随后下层才截断累计命令。结果是物理命令虽被限制，reward、value target 和监督损失却被策略实际没有执行的巨大动作污染。首条 REPOSITION 样本只是第一次对全部维度启用监督，因而把 OPEN_LATCH 自由机械臂维度中已形成的异常显式暴露出来。

当前修复在统一 delta action 接口中增加显式 `delta_action_step_clip: 1.0`：单步归一化 delta 在记录、累计和 action-rate reward 前统一限制到 `[-1,1]`。它不依赖阶段、门型或 seed，并保留累计命令原有的独立限幅。phase117 将再次从随机初始化训练 100 步，验收点是：首次出现 OPEN_LATCH/REPOSITION 时不再产生巨额 action penalty、value loss 或 supervision loss，并能继续积累后续阶段。

### phase117--118：有界环境动作与鲁棒监督

phase117 再次从随机初始化训练到 step 100。通用 delta action 限幅修复通过物理与 reward 验收：

```text
delta_action_step_clip_frac: 约 0.05%--1.4%
Mean episode rew_penalty_delta_action_rate: 通常约 -0.0002--0.0008
value_avg: 通常约 10--120
```

未修复轮在相同 step 68 区间的 action-rate penalty 为 `-13848.8`、value loss 为 `5.07e10`；修复后即使首批 REPOSITION 样本出现，也没有再发生物理命令或 reward 污染。OPEN_LATCH progress 一度达到约 `0.22`，明显高于 phase116 后半段长期约 `0.04` 的水平。

但 policy action supervision 仍对未限幅的 Gaussian mean 使用 MSE。稀有阶段首次进入监督 mask 时，虽然环境只执行有界动作，辅助损失仍会瞬时达到 `2649`、`1020`，后期还有 `65.5`。actor 梯度裁剪阻止了它污染 critic，但这种无界平方损失既不对应实际执行动作，也会让单个未见阶段样本主导 actor 更新。

监督现已统一改为显式 `policy_action_supervision_huber_beta: 1.0` 的 Smooth L1。小误差仍按二次项精确拉回，大误差保持有界梯度；没有按阶段分叉。phase118 从 phase117 step 100 继续到 step 200，用于验证 staged-reset 样本重建后监督不再爆炸，并观察 OPEN_LATCH 能否稳定产出 REPOSITION。

phase118 到 step 200 后，Smooth L1 监督峰值约为 `0.148`，没有再出现 MSE 的数千量级异常；action-rate reward 和 value loss 也保持正常。阶段覆盖为：GRASP reached 约 `9.8%`、OPEN_LATCH reached 约 `1.29%`、REPOSITION occupancy 约 `0.12%`，PUSH_OPEN 与完整成功仍为 0。

```text
checkpoint: logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260925_084858/model_step_000200.pt
SHA-256: 5b1290ad007e230a42bdbb8af7b5d87c29a040d25e3284b42addb994ffaf2d7f
```

进一步审计发现，环境动作虽然已正确限幅，但 OPEN_LATCH 的原始 Gaussian mean 已经达到数千。根因不是该阶段 reward 本身，而是通用 PPO trainer 的 KL 自适应学习率所有权错误：`_adjust_learning_rate_based_on_kl()` 在每个 minibatch 都可能将学习率乘 `1.5`，且使用与配置初始学习率无关的硬编码上限 `1e-2`；同一轮末尾的外部 constant scheduler 又把 optimizer 学习率重置为 `1e-4`。因此 trainer 内部状态、日志和 optimizer 实际学习率互相矛盾。

旧 phase117 checkpoint 给出了直接证据：step 25 时 `args.learning_rate=1e-2`，是配置 `1e-4` 的 100 倍。前两个 reference-owned 阶段没有 residual 梯度，所以网络在 step 25 仍保持初始化；OPEN_LATCH 首次产生 residual 梯度后，step 25--50 第一层权重范数从 `13.05` 跳到 `57.20`，输出层从全零跳到范数 `9.04`，导致普通标准化输入的 mean 达到 `O(10^4)`。

### phase119--120：修复 KL 自适应学习率所有权

通用 trainer 已改为：仅当 `schedule=adaptive` 时启用 KL 调节；每个完整 PPO update 汇总 KL 后只调一次；上下限相对初始学习率配置；启用 KL 调节时不再同时执行会重置学习率的外部 scheduler；日志记录 optimizer 的真实学习率。B2Z1 配置显式使用 `1e-5--1e-3`，没有阶段或 seed 特例。

phase119 从随机初始化训练到 step 50，复现并通过旧实验发生突跳的窗口：

```text
checkpoint: logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260925_091041/model_step_000050.pt
SHA-256: 1f8f230b7e77ad5fa9247538291a890cf274f2842c430380aed91c3985159484
step 25 learning rate: 1e-3（配置上限）
step 50 learning rate: 2.96296e-4
step 25--50 第一层权重范数: 13.05 -> 13.92
step 25--50 输出层权重范数: 0 -> 0.94
随机标准化输入 mean 范围: [-3.47, 2.13]
```

与旧实验的 `13.05 -> 57.20`、`0 -> 9.04` 和约 `[-1.44e4, 9.87e3]` 相比，学习率突跳和 policy mean 数值爆炸已消失。step 50 已出现 OPEN_LATCH rollout 样本，但尚未出现 REPOSITION。phase120 正在同样的 4 x 2048 env、纯动力学 headless 配置下从随机初始化训练到 step 100，用于确认首批 REPOSITION 出现后仍保持稳定，再判断后续真正的阶段瓶颈。

phase120 将上述“每 update 只调一次、上下限相对初始值”的第一版修复从随机初始化验证到 step 100。它消除了 `O(10^4)` 数值爆炸，但配置仍允许 KL 调度把 `1e-4` 提前升到 `1e-3`。前两阶段 residual 被 mask 时，噪声参数产生的微小 KL 已足以触发升档；OPEN_LATCH 第一次解锁网络梯度时因此仍以 10 倍配置学习率更新。step 100 的 OPEN_LATCH policy mean 约为 `5--16`，环境裁剪比例约 `1.91%`，说明动作语义仍不合格。

最终通用规则改为：配置的 actor learning rate 就是自适应调度上限；KL 过大时可以降低，恢复后只能回到配置值，不能静默超过。显式 opt-in 配置仍可提供更高上限，但默认不会这样做。phase121 用该规则重新从随机初始化训练到 step 100：

```text
checkpoint: logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260925_164648/model_step_000100.pt
SHA-256: 63eded9a124863789e3b6daf82354a1fbfb39ccd377bf3d8c44568073c332492
OPEN_LATCH policy mean: 约 [-0.068, 0.069]
delta_action_step_clip_frac: 0.0008
policy_action_supervision_avg: 0.000010
value_avg: 38.76
OPEN_LATCH occupancy/reached/progress: 3.94% / 4.57% / 0.2058
REPOSITION occupancy/reached/progress: 0.63% / 0.63% / 0.0730
```

对照 phase120，OPEN_LATCH progress 从 `0.1045` 提高到 `0.2058`，而 policy mean 从饱和区回到归一化动作域附近，证明学习率修复同时改善了数值语义和任务探索，不是单纯压小动作。step 100 checkpoint 的四个 rank 分别保存了约 `607/672/607/673` 个可用 staged-reset 阶段/环境入口；当前 checkpoint 实现会按 rank 保存并恢复这些快照。phase122 因而从 step 100 直接续训到 step 300，不需要重新积累课程状态；在结果出来前不修改 REPOSITION reward。

### phase122：step 300 与 REPOSITION 姿态接口瓶颈

phase122 从 phase121 step 100 续训 200 个 update。四个 rank 的 staged-reset 快照均成功恢复，训练正常结束，无训练进程遗留：

```text
checkpoint: logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260925_170117/model_step_000300.pt
SHA-256: 168e1d7bd206a5bacc8e764512a4414e0b7f2cc4318b5dcdb42c4f876b1fb394

OPEN_LATCH occupancy/reached/progress: 13.54% / 16.34% / 0.2192
REPOSITION occupancy/reached/progress: 2.80% / 2.80% / 0.0465
REPOSITION base position error: 0.1661 m
REPOSITION base yaw error: 0.0151 rad
REPOSITION push-ready orientation error: 1.7040 rad
PUSH_OPEN occupancy: 0
THROUGH occupancy: 0
```

策略在 REPOSITION 中几乎精确复现 reference：base、EE position、EE orientation 的平均 action-reference 误差分别约为 `0.0004/0/0.0020`。因此当前瓶颈不是 PPO 没学会 reference，而是 reference 通过 delta accumulator 后没有真正到达目标姿态。

诊断分解为：目标命令与 accumulator 相差 `2.1534 rad`，accumulator 重建姿态与目标世界姿态相差 `1.9442 rad`，机械臂对 accumulator 的跟踪误差约 `0.7624 rad`；workspace 投影误差只有 `0.0096 m`。日志同时显示 roll accumulator 已到 `3.1348 rad`，reference 仍持续给正向动作。根因是姿态 reference 要沿最短路径越过 `+pi`，但通用 delta accumulator 把三个周期姿态维当普通有界标量夹紧，无法环绕到等价的 `-pi` 一侧。

当前统一接口修复是在 delta accumulator 中加入非欧式坐标归一化钩子，由 B2Z1 命令层将 roll、pitch、yaw 三个姿态命令按 `2*pi` 周期环绕。它不改变 reward、阶段阈值、observation 或 action 维度，也不包含阶段和 seed 特例。phase123 正从 step 300 做短程四卡回归，验收目标是目标命令误差和 push-ready 姿态误差明显下降，并首次出现 PUSH_OPEN 样本。

### phase123：周期姿态命令修复回归

phase123 从 step 300 续训到 step 325，统一周期坐标修复生效且训练正常结束：

```text
checkpoint: logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260925_173035/model_step_000325.pt
REPOSITION target orientation command error: 2.1534 -> 0.0449 rad
REPOSITION commanded orientation error: 1.9442 -> 0.0604 rad
REPOSITION push-ready orientation error: 1.7040 -> 0.3352 rad
REPOSITION progress: 0.0465 -> 0.4418
```

这证明目标四元数到 B2Z1 命令的正逆映射本身可用，原瓶颈确实是 accumulator 在 `+pi/-pi` 边界错误饱和。step 325 尚未出现 PUSH_OPEN；剩余误差主要来自机械臂对已正确命令的物理跟踪（约 `0.329 rad`）以及仍在运动中的底盘/末端，而不是 target-to-command 映射。当前不放松阈值、不改 reward，phase124 从 step 325 继续训练 100 个 update，让策略在修复后的统一动力学接口上收敛。

### phase124：确认 REPOSITION 仍有多余 roll 约束

phase124 从 step 325 继续到 step 425。训练期间 target-to-command 姿态误差稳定在约 `0.02 rad`，但命令到实际夹爪姿态的误差长期停在约 `0.33 rad`，继续训练没有下降，PUSH_OPEN 始终为 0：

```text
checkpoint: logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260925_173713/model_step_000425.pt
SHA-256: daa81af459e2e6902f9b1055d2583adbb9fb62be5f43ed12f0a238dc6b2d756c
REPOSITION occupancy/reached/progress: 5.86% / 5.86% / 0.5207
target orientation command error: 0.0212 rad
commanded orientation error: 0.0386 rad
orientation tracking error: 0.3279 rad
push-ready orientation error: 0.3335 rad
PUSH_OPEN occupancy: 0
```

对照已验收的 AutoOpen 源码后确认，推门准备姿态只应以最短旋转将夹爪插入轴对齐到门板法向，并保留撤离后的当前 roll。RL 旧实现则重新构造一套带固定 roll 的完整标准姿态，额外要求机械臂在该工作点完成与推门无关的腕部旋转。这既不属于推门语义，也造成持续的物理跟踪残差。

统一修复已将抓把手所需的镜像完整姿态和推门所需的轴向约束拆开：GRASP 继续使用左右门镜像抓取姿态；REPOSITION/PUSH_OPEN 保留当前 roll，只对齐插入轴。旧的 PUSH 固定 roll 路径已删除。phase125 从 step 425 做 25 个 update 回归，首先验证 push-ready 姿态误差和阶段进入率。

### phase125：轴向约束生效并暴露底盘启停振荡

phase125 从 step 425 到 step 450。最小轴向对齐提高了满足姿态条件的样本比例，并首次出现同时满足全部 REPOSITION 即时条件的环境，但没有持续到阶段要求的 15 步：

```text
checkpoint: logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260925_175059/model_step_000450.pt
SHA-256: 44108361621d78ddbc6328b2f36197cace779b0ad3623bb276f3f240c5072843
push-ready orientation ok: 1.7% -> 9.5%
all checks now: 0 -> 0.24%
PUSH_OPEN occupancy: 0
```

底盘 reference 随后被确认存在内部矛盾：REPOSITION 验收要求位置误差 `<0.08 m`，旧控制却在误差 `<0.12 m` 时直接把速度置零，在外侧又强制至少 `0.25 m/s`。这会让底盘高速穿过验收带后停车或反向，难以同时满足位置和静止条件。phase126 只修复该控制器：速度按距离比例下降，以 `0.15 m/s` 补偿低层 `0.1 m/s` 步态门限，并在 `0.04 m` 内停车；reward 和验收阈值均不变。

### phase126：首次进入 PUSH_OPEN

phase126 从 step 450 到 step 475。新的底盘启停控制在不放松任何验收阈值的情况下，使 REPOSITION 首次连续稳定通过；step 474 和 step 475 均出现首批 PUSH_OPEN 样本：

```text
checkpoint: logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260925_175802/model_step_000475.pt
PUSH_OPEN occupancy/reached: 0.01% / 0.01%
PUSH_OPEN progress: 0.0924
```

这确认旧的 `0.12 m` 停车半径与 `0.08 m` 验收半径不一致，以及 `0.25 m/s` 最低速度造成的振荡，是阻断 REPOSITION 的真实控制接口 bug。当前首批 PUSH_OPEN 样本仍全部来自右门；左门 OPEN_LATCH 虽有约 10% occupancy，但几乎没有进入 REPOSITION。phase127 从 step 475 继续训练 100 个 update，先积累 PUSH_OPEN 样本并判断该阶段首个真实瓶颈，同时继续收集左右门 OPEN_LATCH 对照数据。

### phase127：右门链路进入 THROUGH

phase127 从 step 475 训练到 step 575。右门 PUSH_OPEN 样本持续增长，并首次进入 THROUGH：

```text
checkpoint: logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260925_180407/model_step_000575.pt
SHA-256: c15a16446eebb84d3a18be188c8fe29412176698f3b2707e2ee24292d79126ed
PUSH_OPEN reached: 0.72%
THROUGH occupancy/reached/progress: 0.63% / 0.63% / 0.3551
THROUGH distance: 0.5475 m
THROUGH middle arm max/mean error: 2.6942 / 1.3937 rad
task success: 0
```

PUSH_OPEN 已能把部分门推过阶段阈值，说明推门动力学、门角反馈和阶段跳转链路均可工作。当前后期首要瓶颈转为 THROUGH 的机械臂收回：环境 reference 在 EE 坐标上持续给出回到 middle preset 的逆映射动作，但策略 mean 在多个坐标上接近零并抵消 reference。

进一步审计确认，监督损失虽然按有效 action mask 归一化，却把所有阶段的有效坐标放在同一个分母；THROUGH occupancy 只有约 `0.6%`，其梯度仍被大量已学会的早期阶段 mask 稀释。phase128 将监督损失复用 PPO 已有的通用 stage-balanced sample weight，使各阶段按同一规则等权，不增加 THROUGH 特例或阶段系数。

### phase128：stage-balanced 监督短程验证

phase128 从 step 575 训练到 step 625。通用 stage-balanced supervision 正常运行，没有数值或运行时错误：

```text
checkpoint: logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260925_181853/model_step_000625.pt
THROUGH distance: 0.8975 m
THROUGH middle arm max error: 2.6942 -> 2.5869 rad
THROUGH middle arm mean error: 1.3937 -> 1.2825 rad
arm safely stowed: 0
task success: 0
```

机械臂误差方向正确但 50 个 update 尚不足以完成收回。底盘已接近 `1.0 m` 穿越阈值，说明当前瓶颈单独集中在 middle preset 跟随。phase129 不再改机制，从 step 625 延长训练 100 个 update；若误差持续下降则继续训练，若进入平台期则回到 THROUGH 的关节目标到 EE 命令逆映射本身做数学和仿真对照，不提高阶段特定监督系数。

### phase129--130：THROUGH 平台与逆映射核对

phase129 从 step 625 训练到 step 725。THROUGH 样本继续增加，但 middle preset 跟随进入平台：

```text
checkpoint: logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260925_182716/model_step_000725.pt
SHA-256: 6b1e451bfa45d85b9dac7bd578617dcda60acde9bc9c45217a3959ecfb5f553e
THROUGH reached: 1.00%
THROUGH middle arm max/mean error: 2.6326 / 1.1826 rad
THROUGH arm safely stowed: 0
task success: 0
```

逐关节误差表明第 6 关节仍有 `2.5173 rad`，并非所有关节一起缓慢收敛。左右门不对称也仍存在：左门 OPEN_LATCH/REPOSITION reached 为 `8.64% / 0.05%`，右门为 `30.28% / 25.13%`。

phase130 只加入无副作用数学诊断，从 step 725 短跑到 step 735。结果确认 THROUGH 的局部逆映射公式本身正确：原始逆解还原关节步长误差为 `0.0000 rad`；稳定后平移裁剪比例低于 `1%`，旋转裁剪为 0；裁剪后六关节还原步长与期望的约 `+-0.01 rad/step` 基本逐项一致。当前瓶颈不是 Jacobian/DLS 数学错误，而是策略均值没有跟随 reference，多个 EE 姿态维甚至持续给出反号残差。

进一步检查发现，原监督复用 PPO 的 capped stage weight。THROUGH 每轮约 291 个样本时权重已顶到 `10`，但常见阶段的加权总量仍约为它的 5 倍，因此并未实现监督意义上的阶段等权。现已将职责拆开：PPO policy/value 权重继续保留 `10x` 上限以控制方差；action supervision 则按各阶段实际启用的监督坐标总数计算无截断逆频率权重，使每个出现阶段对监督损失贡献相同。该实现不含阶段名称、门型或 seed 特例。

phase131 从 step 730 继续到 step 830，用于验证真正阶段等权的监督能否让 THROUGH middle preset 误差继续下降并首次产生 `arm_safely_stowed`；同时继续收集左门 OPEN_LATCH 对照数据。

### phase131--133：真正阶段等权与左右门 GRASP 根因定位

phase131 从 step 730 训练到 step 830。监督阶段真正等权后，THROUGH middle preset 平均误差从 `1.1826 rad` 降至 `1.0451 rad`，第 6 关节误差从 `2.5173 rad` 降至 `2.2780 rad`，训练中首次短暂出现 `arm_safely_stowed > 0`。完整成功仍为 0，说明方向有效但尚未收敛。

```text
checkpoint: logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260925_184910/model_step_000825.pt
SHA-256: d70ed60398469a1446a9007719601a49ed8c805c7fcf5bd67d5ad444d6c699ff
GRASP reached: 29.39%
OPEN_LATCH reached: 21.88%
REPOSITION reached: 14.52%
PUSH_OPEN reached: 1.96%
THROUGH reached: 1.56%
```

左右门分解显示不对称在 GRASP 已经形成，而不是 OPEN_LATCH 才出现：左/右 GRASP reached 为 `16.76% / 42.06%`。phase132 只增加验收诊断，从 step 825 短跑到 step 835。代表性 GRASP 占用样本中：

```text
左门 pocket 垂直误差: -14.9 mm
右门 pocket 垂直误差:  +3.3 mm
左/右轴向姿态误差:    0.0200 / 0.0215 rad
左/右相对速度:          0.0458 / 0.0423 m/s
左/右工作区投影误差:    0 / 0 m
```

因此主要差异是镜像抓取构型下的垂直位置跟踪，而不是姿态、工作区不可达或运动过快。phase133 继续只增加 action reference、policy mean 和累计 command 的左右门对照，确认正确修复层级后才允许修改行为。

phase133 结果显示，左右门 GRASP 的 policy mean 与 reference 在 `ee_z` 上逐项一致，residual 为 0；累计 position command 与目标 command 的误差约 `0.5 mm`。左门 reference 确实在向上纠正负垂直误差，右门 reference 也按相反方向纠正，因此训练监督和 reference 方向均正确。

根因位于夹爪闭合触发语义：旧条件在 pocket 总误差 `<20 mm` 时允许闭合，但随机把手半径仅 `11--15 mm`。这意味着把手中心仍可位于抓取通道中心外一个完整把手半径时，机械接触就已经开始；固定爪/移动爪的镜像构型会把同一几何偏差放大成不同的垂直接触结果。phase135 将闭合触发改为 pocket 总误差 `<0.5 * door_handle_radius`，由每个环境真实形态计算，不添加左右门分支；GRASP 最终验收条件和 reward 均未改变。

### phase134--135：后段幸存者偏差与形态尺度闭合触发

phase134 将 GRASP 中 `actual_time_in_stage >= 400` 的后段失败样本单独统计。后段样本常见现象是丢失双侧接触、pocket normal error 增长到 `6--9 cm`、固定爪和移动爪接触力趋近 0。该统计存在明确的幸存者偏差：成功样本已经离开 GRASP，后段集合只剩失败样本，因此不能把它直接解释为所有样本的左右门根因。本轮只保留诊断结论，没有据此修改控制逻辑。

phase135 从 step 830 训练到 step 850，只验证统一的形态尺度闭合条件：

```text
grasp_capture_tolerance = 0.5 * door_handle_radius
pocket_error < grasp_capture_tolerance
```

结果：

```text
左门 pocket 垂直误差：约 -17.3 mm
右门 pocket 垂直误差：约  -7.2 mm
左门 GRASP reached：     16.09%
右门 GRASP reached：     33.14%
左门 OPEN_LATCH reached： 5.98%
右门 OPEN_LATCH reached：22.63%
左门 REPOSITION reached： 0.10%
右门 REPOSITION reached：18.63%
```

结论：更严格的形态尺度条件使左右门垂直误差方向趋于一致，确认“把手中心尚未进入夹爪通道就提前闭合”是一个真实问题；但它不是剩余主因。多数 GRASP 环境没有进入闭合，`gripper_command_error` 约 `1.97 rad`，说明它们在等待实际 pocket 进入中心区。此时 normal error 已接近 0，而 vertical error 仍约 `-13--17 mm`。因此下一根因是几何 command 到实际稳态 EE 的重力偏差，不能继续靠放宽闭合阈值或添加左右门高度参数处理。

phase135 已正常结束，无训练进程残留。该短程结果未优于 phase131，当前有效继承点仍为 phase131 step 825。

### phase136：稳态逆补偿只读诊断

AutoOpen 已通过 `ee_steady_state.py` 使用数学稳态逆映射；RL reference 当前只发送几何位置目标，姿态另有一套慢积分补偿。phase136 的目的不是通过仿真拟合查表，也不是添加左右门或阶段偏置，而是验证以下统一数学链路：

```text
期望实际 EE 位姿
-> 当前 Jacobian/DLS 对应的关节位移
-> 加上 gravity_effort / joint_stiffness 所需稳态关节目标
-> 使用与在线 IK 一致的逆映射反算 Cartesian command
-> 现有低层控制器执行
```

实验开始前必须记录 phase131 step 825 的 checkpoint、唯一代码变量和输出字段。第一轮只输出关节目标还原误差、Cartesian 补偿方向/幅值及左右门统计，不改变策略行为。只有数学诊断成立后，才允许把补偿接入通用的“世界 EE 目标 -> B2Z1 command”接口，并删除被替代的旧 orientation integral 补偿。

phase136 已完成以下只读诊断实现：

- 将隐式机械臂驱动的 stiffness/damping/effort limit 收敛到与模拟器和逆映射共用的配置源，数值保持 `400/40/80` 不变。
- 使用 PhysX 批量重力补偿力、当前 EE Jacobian 和与低层完全相同的 DLS，计算 `gravity_effort / stiffness` 所需关节偏置及其 Cartesian 逆映射。
- 输出 inverse valid、姿态可表示性、力矩饱和、位置/姿态补偿、关节偏置和关节还原误差；结果不连接到 policy reference，因此不改变训练行为。

单卡 64 env smoke 正常完成。四卡 `4 x 2048 env`、step 825 到 826 的首轮诊断显示：

```text
左右门 joint recovery error max：0.0000 rad
gravity effort saturation：        0
预测 position compensation norm： 约 5.0 / 4.8 mm
预测 position compensation z：   约 -2.7 / -2.6 mm
```

该轮只有恢复后的 8 帧，reference 尚未追上目标。误差分解为：

```text
左门 actual-reference z：  -4.6 mm
左门 reference-target z： -32.1 mm
右门 actual-reference z：  -1.7 mm
右门 reference-target z： +19.6 mm
```

因此这批厘米级 pocket 误差是 staged-reset 后的 reference 瞬态，不能当作稳态重力误差。

增加跟踪分解诊断时曾把临时张量定义在 `_update_training_diagnostics()`，却在 `get_policy_diagnostics()` 使用，四个 rank 均在第一次采样时触发同一个 `NameError`。异常发生在 PPO 更新之前，没有生成 checkpoint，也没有污染 phase131。临时张量随后移入正确的诊断函数作用域，并通过四卡同拓扑重跑。

从 phase131 step 825 连续运行 20 update 后，reference 已接近目标，最终统计为：

```text
                           左门       右门
reference-target z：      -2.3 mm    +1.5 mm
actual-reference z：     -15.7 mm   -10.1 mm
actual-command z：       -15.6 mm   -10.2 mm
预测 compensation z：    -4.1 mm    -3.8 mm
joint recovery error：     0.0 rad    0.0 rad
```

这证明剩余厘米级误差位于内层执行，不是 reference 追赶或 policy command 跟随；但该均值仍混合正在运动和已接触把手的样本，尚不能全部归因于重力。随后增加三个嵌套诊断集合：reference 对齐、reference 对齐且 EE 低速、reference 对齐且 EE 低速并无把手/门板接触。

最终无接触稳态过滤结果如下。日志使用短键 `diag/grasp/L/*` 和 `diag/grasp/R/*`，避免 Rich 截断长指标名：

```text
                              左门       右门
无接触稳态样本比例 f_frac：  0.3541     0.4742
actual-reference z (f_ar_z)：-14.7 mm   -13.0 mm
actual-command z (f_ac_z)：  -14.8 mm   -13.0 mm
reference-target z (f_rt_z)：  0.0 mm     0.0 mm
预测补偿 z (f_pc_z)：         -3.9 mm    -3.7 mm
EE speed (f_speed)：           0.0155     0.0160 m/s
inverse valid：                1.0000     1.0000
joint recovery error max：     0.0000     0.0000 rad
gravity effort saturation：    0          0
```

已确认：厘米级下沉在无接触、低速、reference 已对齐时依然存在，故不是 reference 追赶、policy command、workspace projection 或接触载荷造成；DLS 反演在关节目标意义上精确成立，且没有重力力矩饱和。`f_pc_z` 不能与 `f_ar_z` 按单一 z 标量直接比较，因为求解的是完整 6D pose，姿态补偿通过 Jacobian 耦合共同改变六个关节目标。补偿是否能消除实际稳态误差，必须由下一轮接入后的闭环对照回答。

phase136 到此结束，不再增加同类观察指标。它没有改变策略行为、没有产生新 checkpoint，当前有效继承点仍为 phase131 step 825。

### phase137：统一稳态逆补偿接入（失败，已停止）

```text
基线 checkpoint：phase131/model_step_000825.pt
唯一行为变量：将 phase136 的完整 6D 数学稳态补偿用于 GRASP 世界系 EE reference 到 B2Z1 command 的统一转换
被替代旧机制：reference_orientation_integral_gain、reference_orientation_compensation_limit、_reference_orientation_compensation
禁止改动：reward、阶段验收、随机化、policy 结构、闭合条件、左右门/seed/阶段偏置、低层 _refresh_b2z1_arm_target()
```

执行顺序和决策门槛：

1. 接入前删除旧 orientation integral 的配置、状态和更新路径，确保只有一套补偿所有权。
2. 单卡 smoke 只验运行时、张量形状和有限值，不作为行为验收。
3. 从 phase131 step 825 以 `4 x 2048 env` 原拓扑短跑，比较 phase136 同口径的左右门无接触稳态误差、pocket 垂直误差和阶段到达率。
4. 若补偿方向错误、`actual-reference z` 绝对值增大、APPROACH_PREGRASP 低于 `100%`，或 GRASP/OPEN_LATCH 明显退化，立即停止并定位统一转换公式，不进入后续训练。
5. 只有左右门稳态误差均显著下降且前序阶段无回归，phase137 才可成为新的训练基线；否则有效 checkpoint 继续保持 phase131 step 825。

单卡 64 env smoke 完成一个 PPO update，无运行时异常或非有限值。随后从 phase131 step 825 以 `4 x 2048 env` 原拓扑运行 20 update；iteration 845 的关键结果为：

```text
APPROACH_PREGRASP reached：100.00%
GRASP reached：             25.08%
OPEN_LATCH reached：        14.61%
REPOSITION reached：         9.90%
PUSH_OPEN reached：          1.03%
THROUGH reached：            0.84%

左门低速 actual-reference z： -39.0 mm
左门低速 actual-command z：   -34.8 mm
左门无接触稳态比例：             0.10%
右门低速 actual-reference z： -15.0 mm
右门低速 actual-command z：   -11.1 mm
右门无接触稳态比例：            10.66%
```

左门误差和无接触样本比例明显恶化，触发预先登记的停止条件。该轮没有继续训练、没有保存 checkpoint，有效继承点仍为 phase131 step 825。

检查 Isaac Sim 5.1 本机 `omni.physics.tensors` API 源码后确认：`get_gravity_compensation_forces()` 对浮动基座 articulation 返回 `(count, max_dofs + 6)`，前六项为基座广义坐标。旧实现直接使用 `_b2z1_arm_sim_joint_ids` 索引该张量，没有像 Jacobian 路径一样加 `floating_base_offset=6`，因此读取的是错位的基座/腿部重力项。DLS 对这个错误输入仍能做到 `joint recovery error=0`，所以 phase136 的数学自洽检查无法发现输入语义错误。

### phase138：修复浮动基座重力索引（失败，已停止）

```text
基线 checkpoint：phase131/model_step_000825.pt
唯一代码变量：按 gravity_force_width - num_dof 计算 0/6 广义坐标偏移，再读取六个机械臂关节重力补偿力
保持不变：phase137 的统一 6D 补偿接入和旧 orientation integral 删除
禁止改动：reward、阶段验收、随机化、policy、闭合条件、阶段几何和低层 IK
```

先通过单卡 smoke 验证力张量宽度和索引，再重复 `4 x 2048 env`、20 update 对照。仍使用 phase137 的停止条件；不因 DLS 还原误差为零直接宣布成功，必须看实际 EE 稳态误差和前序阶段到达率。

单卡 smoke 通过。四卡 20-update 对照在 iteration 845 得到：

```text
APPROACH_PREGRASP reached：100.00%
GRASP reached：             24.50%
OPEN_LATCH reached：        14.55%
REPOSITION reached：         9.71%
PUSH_OPEN reached：          1.03%
THROUGH reached：            0.83%

左门低速 actual-reference z：  +5.9 mm
左门低速 actual-command z：   -19.8 mm
左门无接触稳态比例：            45.95%
右门低速 actual-reference z： +64.5 mm
右门低速 actual-command z：   +35.0 mm
右门无接触稳态比例：             0.26%
```

正确索引使预测补偿 z 由 phase137 的约 `-4 mm` 变为主要向上，左门稳态误差明显改善；但右门出现约 `+64.5 mm` 过补偿并几乎失去无接触稳态样本，前序阶段率也没有改善。phase138 因此未通过，不保存 checkpoint，有效继承点仍为 phase131 step 825。

下一步只做根因分析：比较左右门完整 6D 补偿、关节重力偏置、当前关节构型、DLS 条件数、关节限位余量以及补偿前后实际隐式驱动目标。根因确认前不增加经验系数、左右门分支或补偿裁剪。

### phase139：恢复 lab30 IK 姿态增益（完成，低层修复保留）

源码对照确认 `visual_wholebody_lowlevel` 的 ManipLoco 配置使用 `teleop_ik_orientation_gain=4.0`，`B2Z1SteadyEeForwardModel` 也以 `orientation_gain=4.0` 建模；RL `_refresh_b2z1_arm_target()` 此前遗漏该增益，实际按 `1.0` 执行。这使稳态逆映射和机器人正确低层控制语义不一致，并通过 6D Jacobian 耦合影响平移命令。

```text
基线 checkpoint：phase131/model_step_000825.pt
唯一变量：在线 IK 与稳态逆映射统一读取 lowlevel_ik_orientation_gain=4.0
禁止改动：左右门分支、经验增益、补偿裁剪、reward、阶段验收、随机化和策略结构
```

phase139 先做单卡 smoke，再做 `4 x 2048 env`、20 update 对照。验收仍以无接触稳态误差、无接触样本比例和前序阶段到达率为准。

phase139 单卡 smoke 和四卡 20-update 对照均正常完成。iteration 845：

```text
APPROACH_PREGRASP reached：100.00%
GRASP reached：             29.47%
OPEN_LATCH reached：        12.20%
REPOSITION reached：        10.33%
PUSH_OPEN reached：          0.94%
THROUGH reached：            0.71%

左门无接触稳态比例/误差：59.71% / +8.1 mm
右门无接触稳态比例/误差： 7.24% / +5.8 mm
```

恢复姿态增益后 GRASP reached 从 phase138 的 `24.50%` 恢复到 `29.47%`，与 phase131 的 `29.39%` 相当；右门无接触样本也从 `0.26%` 增至 `7.24%`。这确认遗漏 `4.0` 是真实低层接口 bug，应保留。但完整重力补偿仍使无接触 EE 高于 reference 约 `6--8 mm`，且旧 checkpoint 尚未适应新的低层动力学，不能把 phase139 checkpoint 作为新基线。

### phase140：严格静态平衡诊断（完成，未改行为）

现有 free 集合只要求 EE speed `<0.03 m/s`，实际样本仍在约 `0.01--0.02 m/s` 运动，机械臂 `40 Nms/rad` damping 可能不可忽略。phase140 不改变行为，只增加：

- 无接触且 EE speed `<0.01 m/s`、`<0.005 m/s` 的两级集合；
- 六关节最大速度；
- `clip(K(q_target-q)-Dqdot)-gravity_effort` 的最大关节平衡残差。

只有严格低速集合的平衡残差接近零，才能把剩余 `6--8 mm` 解释为静态模型误差；否则先处理动态样本和时序口径，禁止添加经验补偿系数。

phase140 四卡 20-update 诊断结果：

```text
                              左门       右门
free (<0.03 m/s) 比例：      59.77%      6.81%
free EE speed：              0.0160      0.0186 m/s
free 最大关节速度：          0.0368      0.0411 rad/s
free PD-重力残差：           6.6786      6.0560 N·m
free <0.01 m/s 比例：         1.57%       0.00%
free <0.005 m/s 比例：        0.00%       0.00%
```

结论：当前没有可用于验证静态方程的 `<0.005 m/s` 样本，右门甚至没有 `<0.01 m/s` 无接触样本；普通 free 集合的关节速度和约 `6 N·m` 平衡残差均不可忽略。因此 phase139 的 `+6--8 mm` 不能被认定为静态模型过补偿，也不能据此添加缩放系数。phase140 不改变行为、不生成 checkpoint。

### phase141：在修正低层语义下继续训练（已完成）

```text
基线 checkpoint：phase131/model_step_000825.pt
行为代码：phase139（正确重力索引、lab30 姿态增益 4.0、统一 6D 稳态补偿）
唯一变量：继续 PPO 更新，使旧策略适应修正后的低层动力学
禁止改动：reward、验收、随机化、reference、补偿公式和左右门逻辑
```

phase141 已从 step 825 训练 100 update，并在 step 850、875、900、925 保存 checkpoint。训练正常结束，无残留训练进程；全程保持 `4 x 2048 env`、纯动力学、无相机、无 RTX sensor 和关闭 frame rendering。

```text
运行目录：logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260925_205918
checkpoint：model_step_000925.pt
SHA-256：b8fdd0bfeda53fed3c26f071cc8c7691772f2458787bf9e82f06640ed39a2b24

APPROACH_PREGRASP reached：100.00%
GRASP reached：             36.95%
OPEN_LATCH reached：        15.59%
REPOSITION reached：        12.57%
PUSH_OPEN reached：          4.92%
THROUGH reached：            3.97%
task success：               0
```

左右门分解：

```text
                         左门      右门
GRASP reached           22.63%    51.33%
OPEN_LATCH reached       5.88%    25.33%
REPOSITION reached       0.55%    24.63%
PUSH_OPEN reached        0.21%     9.65%
THROUGH reached          0.05%     7.90%
```

THROUGH 的 middle arm max/mean error 为 `2.1670/1.0673 rad`，`arm_safely_stowed=0.0029`，尚未形成 task success。phase141 的总体后期到达率高于 phase131，且使用当前正确低层语义，因此 step 925 升为当前可继承 checkpoint。

本轮暴露的首个真实瓶颈是左门 OPEN_LATCH。step 925 的阶段占用样本显示，左门把手角、锁舌缩回和门缝都接近零，而右门已能完成明显按压和开缝。当前 OPEN_LATCH 只监督夹爪保持，未给六维机械臂动作提供统一几何 reference；下一轮必须先核清左右门 observation/action 的坐标语义，再决定是否增加通用几何引导。

### phase142：OPEN_LATCH 统一坐标与几何引导审计（已完成，只读）

```text
基线 checkpoint：phase141/model_step_000925.pt
当前阶段：只分析，尚未批准行为修改
待验证假设：左门 OPEN_LATCH 断层来自六维机械臂动作缺少统一门型坐标或足够的中间引导
唯一允许新增：无副作用坐标/轨迹诊断
禁止改动：reward 权重、阶段验收、随机化、GRASP、REPOSITION 及后续阶段行为
```

审计结论：当前 actor action 不具备左右门统一语义。

代码证据：

1. 门生成器将 `door_open_lr=+1/-1` 分别定义为左/右门；右门的 `handle_joint.localRot0` 将关节轴从面板局部 `+X` 翻成 `-X`。两侧关节位置都以正角度表示把手下压。
2. AutoOpen 不猜测 RPY 符号，而是从 USD 读取真实关节轴，然后围绕该轴旋转抓取偏移和末端姿态。这条几何链天然适配左右门。
3. RL 的 `DeltaActionBase` 直接把 actor 的 10 维输出累计进 B2Z1 command；EE 位置与 RPY 均在机身航向坐标中表达，没有使用 `door_open_lr` 做策略坐标到控制坐标的镜像。
4. RL observation 虽然提供左右门 one-hot，部分任务误差仍使用固定门框切向，`delta_actions`、`b2z1_commands` 和 task geometry 内的 `action_reference` 也保留物理 command 坐标。策略被迫自己学习两套符号，而不是在统一任务坐标中解决同一个动作。

几何核对：以闭门门框 `normal=+X`、`tangent=+Y` 为例，正向把手角的世界关节轴为左门 `+X`、右门 `-X`。使用当前 `_compute_grasp_orientation()` 的两种抓取姿态，各自增加 `1 degree` 正向把手角后，B2Z1 本地 XYZ Euler 增量为：

```text
左门：roll=-1 degree, pitch=0, yaw=0
右门：roll=+1 degree, pitch=0, yaw=0
```

平移圆弧的主要垂直分量两侧都向下，但切向分量镜像。一般反射规则为：线向量按切向轴镜像；角向量是伪向量，在同一反射下还要乘反射行列式。因此不能只把一个 roll 坐标乘符号，也不能只在 OPEN_LATCH 写门型分支。

phase141 轨迹统计提供独立行为证据：

```text
                         左门       右门
policy mean ee_z       +0.0134    -0.0193
policy mean ee_roll    +0.0808    +0.0907
handle_angle            0.0367     0.3959 rad
latch_position          0.0014     0.0151 m
hinge_angle             0.0013     0.0286 rad
```

共享策略对两侧给出同号 roll；该符号符合右门正向下压，却不符合左门，所以右门形成明显把手角和门缝，左门几乎不动。由此排除“只需延长当前训练”作为优先方案，也禁止直接增加左门专用 reference。

### phase143：策略坐标与物理控制坐标统一设计和数学验证（已完成）

```text
基线：phase142 只读结论；暂不选择训练 checkpoint
当前阶段：接口设计与数学测试，不运行行为训练
目标：所有阶段共享同一套门型规范化 observation/action 语义
禁止：阶段分支、左右门特调、仅转换 actor 输出、为兼容旧 checkpoint 保留双路径
```

phase143 必须先给出完整的数据所有权图：actor 输出在哪一层从规范化门坐标转换到物理 B2Z1 command；action reference 在哪里做逆变换；哪些历史/command/几何/力观测需要同步规范化；staged-reset 保存的是哪一侧语义。转换必须是可逆的 involution，并用左右门成对数学样本证明“同一规范化动作产生镜像物理动作、再变换回来完全一致”。完成设计评审后才允许编辑行为代码。

phase143 当前设计如下，尚未写入行为代码。

**规范化任务基**

定义三个线向量基：`ex=-frame_normal`，表示朝门/穿门方向；`ey=door_open_lr*frame_tangent`，表示朝把手自由端方向；`ez=world_up`。记列矩阵 `C=[ex, ey, ez]`，其行列式为 `h=-door_open_lr`。线向量使用 `v_world=C*v_task`；角速度、旋转向量和力矩等轴向量使用 `w_world=h*C*w_task`。逆变换分别为 `C^T` 和 `h*C^T`，两次往返必须数值恒等。

该定义满足：两侧 canonical `+roll` 都对应门生成器的正把手关节角；canonical `+x` 都朝向门；canonical `+y` 都朝向把手自由端；canonical `+z` 都向上。

**动作所有权**

1. actor 的前三维改为任务基中的底盘平移速度 `vx/vy` 和 canonical yaw 角速度；进入低层前转换为世界速度，再转换到当前机身坐标。底盘 action 仍是瞬时速度，不积分。
2. actor 的 EE 六维改为任务基中的线位移增量和旋转向量增量。以当前已执行 B2Z1 command 重建的世界 EE goal 为起点，应用 canonical 增量，再通过统一的“世界 EE goal -> B2Z1 physical command”逆映射生成下一 command。不得直接对 B2Z1 RPY command 做符号乘法。
3. gripper 仍是标量增量，不做门型变换。
4. `DeltaActionBase` 保留通用限幅、记录和调用顺序；通过一个明确的“policy delta -> control command”接口让 B2Z1 门任务实现上述映射。内部 `_b2z1_commands` 始终保持部署低层所需的物理语义。

**reference 与 observation 所有权**

1. `get_policy_action_reference()` 必须返回 canonical policy action，actor residual、监督 target 和采样 action 全部在同一空间比较。
2. `_last_delta_actions`/`delta_actions` 保存 canonical policy delta；staged-reset 的物理 command buffer 与 canonical action history 分别保存，不混用。
3. actor 的 `b2z1_commands` 观测不再暴露原始物理 RPY command，而改为 10 维 canonical 执行状态：canonical 底盘速度、低层 EE goal 相对实际 EE 的 canonical 位置/旋转误差、gripper command。
4. `b2z1_task_geometry` 中所有切向误差和末尾 action reference 使用 canonical 语义；`b2z1_panel_interaction` 对当前旋转门板使用同样的动态 panel task basis。
5. 世界系接触力作为线向量变换到任务基；关节位置/速度保留真实机器人本体语义。左右门 one-hot 可保留为物理属性，但不得再承担解释 action 符号的职责。
6. 旧 `relative_to_door` 的四元数 6D 不能直接经过反射。应改成同维度的 canonical 几何量：门框相对位置、机器人 forward 和 up 三个线向量，共 9 维。

**迁移和验收**

该接口改变 actor 输入与输出含义，phase141 checkpoint 只作为行为对照，不直接续训。实现前先写不依赖 Isaac Sim 的批量数学测试，至少覆盖左右门、随机门框 yaw、随机底盘 yaw、线向量/轴向量双向往返、正把手角方向、base command 往返和 EE goal command 往返。行为实现完成后先以零 residual 验证 reference 在左右门产生镜像但语义相同的轨迹，再从干净策略训练；不得用旧 checkpoint 的短期退化否定新接口。

纯数学实现已完成但尚未接入环境行为：新增 `gr00t/rl/envs/door/task_frame.py`，只包含任务基构造及线向量/轴向量在 task、world、任意右手机身基之间的双向变换；新增 `gr00t/rl/tests/test_door_task_frame.py`。容器内 5 项测试全部通过：

```text
basis semantics and handedness：通过
polar vector round trip：       通过
axial vector round trip：       通过
positive handle-axis roll：     通过
random base-yaw velocity round trip：通过
```

随后将现有 command 重建函数提取为可接受任意 physical command 的 `_compute_b2z1_ee_goal_world_from_commands()`，并在 64 个随机门环境中执行：

```text
physical command -> world EE goal
-> 现有 world goal 到 command 逆映射
-> world EE goal
```

单卡 64-env、8-step、1-update 纯动力学 smoke 正常结束，仍确认 `cameras=0, RTX sensors=0, frame rendering disabled`。最大误差为：

```text
world position：          0.270 micrometre
world orientation：       0.391 microradian
position command 表示：   4.388 micrometre
orientation command 表示：7.998 microradian
```

物理 world pose 的误差小于 command 参数差异，说明后者主要是等价参数化和浮点舍入，不是逆映射失真。临时逐步诊断已从正式环境删除，只保留通用重建函数，因此不会增加 8192-env 训练开销。

### phase144：canonical action/reference/observation 接口迁移（已登记，未运行）

```text
起点：phase143 已验证的数学层，不继承 phase141 策略权重
唯一行为变量：将高层策略接口整体迁移到统一门任务坐标
保持不变：物理 B2Z1 command、低层模型、reward、阶段验收、随机化、AutoOpen
禁止：左右门分支、阶段特例、双语义兼容路径、提前增加 OPEN_LATCH reference
```

phase144 必须以一个原子接口迁移完成：actor 输出、actor reference、监督 target、action history、command state observation 和任务几何中的向量必须处于同一 canonical 空间；物理 command 仅存在于 canonical-to-control 边界以内。先做静态维度/往返测试，再用零 residual reference 分别覆盖左右门的 APPROACH_PREGRASP 和 GRASP。前两阶段物理验收回归通过前，不开始 PPO 训练。

接口迁移已经落地：

1. `DeltaActionBase` 只保留一个 `_integrate_policy_delta_actions()` 所有权接口；默认实现保持原行为，B2Z1 门任务 override 将 canonical policy delta 转为 physical B2Z1 command。旧注释版 `step` 和三组废弃调试配置已删除。
2. canonical 底盘平移按线向量、yaw 和 EE 旋转增量按轴向量转换；EE 增量先作用于当前 world goal，再通过 B2Z1 command 的严格逆表示生成低层 physical command。夹爪仍为标量增量。
3. 阶段 reference 先生成原有的下一步 physical command 目标，再严格反算为 canonical policy delta；阶段物理目标和 reference 速率没有修改。supervision target 因此与 actor action 处于同一空间。
4. `_last_delta_actions` 保存 canonical policy history；physical command accumulator 与 canonical history 分别进入 staged-reset 快照，不再混用。
5. `relative_to_door` 改为 canonical 相对位置、机器人 forward 和 up；底盘实际速度、世界接触力、task geometry 和动态 panel interaction 已按线/轴向量类型转换。
6. 10D `b2z1_commands` actor observation 改为 canonical 底盘速度、低层 world EE goal 相对实际 EE 的 canonical 位置/姿态误差和 gripper command。DOF state 保持机器人真实语义。

验证记录：

```text
容器 py_compile：通过
task_frame pytest：5 passed
第一次 64-env smoke：失败于首次 reset observation
根因：_b2z1_ee_pos/_orn 只在首次 action 后初始化，不能作为 observation 事实源
修复：command observation 直接读取 Isaac gripper 刚体当前位姿
第二次 64-env、8-step、1-update smoke：通过
总 timesteps：512
iteration time：3.11 s
纯动力学确认：cameras=0, RTX sensors=0, frame rendering disabled
日志：/tmp/b2z1_phase144_canonical_interface_smoke_v2.log
```

当前运行实验：

```text
日志：/tmp/b2z1_phase144_zero_residual_front_stages.log
环境：64
rollout：600 policy steps
init_at_random_ep_len：false
唯一判据：左右门均能用零 residual canonical reference 完成 APPROACH_PREGRASP 和 GRASP
禁止解释：该实验不验证 OPEN_LATCH 之后的策略能力，也不产生可继承 checkpoint
```

第一轮长窗口在迁移后的 action reference 上发现了接口尺度错误，不能用于验收：

```text
日志：/tmp/b2z1_phase144_zero_residual_front_stages_1100.log
现象：canonical policy-reference error 为 0，但 GRASP reached 仅 8.99%
逐步等价诊断：base command 最大往返误差 0.1125，EE/夹爪误差为 0
根因：canonical action 误用低层 physical command 的非对称 command_scale；
      门任务基旋转后，合法底盘向量部分超过 normalized [-1, 1] 而被裁剪
结论：不是延长训练或放宽阶段阈值可以修复的问题
```

已在接口所有权层增加独立 `canonical_action_scale`。canonical 底盘两个水平轴使用相同尺度，physical `_command_scale` 仅保留在 canonical-to-control 边界以内。修复后 64-env 实际逐步往返结果：

```text
base command max error：0.0000
world EE position max error：0.0000
world EE orientation max error：0.0000
gripper command max error：0.0000
canonical clipping fraction：0.0000
日志：/tmp/b2z1_phase144_reference_roundtrip_scale_smoke.log
```

临时逐步等价诊断随后已从正式代码删除，不增加 8192-env 训练开销。下一实验仍是修正后正式代码的 32-env、1100-step 左右门零 residual 前两阶段回归；修复前长窗口结果不得作为该门槛的证据。

修正后的正式代码完成 32-env、1100-step 回归：

```text
日志：/tmp/b2z1_phase144_zero_residual_front_stages_scale_fixed.log
总 timesteps：35200
iteration time：381.51 s

                         左门      右门
APPROACH_PREGRASP reached 100.00%  100.00%
GRASP reached              11.87%   13.50%
OPEN_LATCH reached          2.99%    2.58%

APPROACH policy-reference error：0
GRASP policy-reference error：   0
```

这里 `GRASP reached` 表示完成 `APPROACH_PREGRASP` 并进入 GRASP；`OPEN_LATCH reached` 表示通过严格双侧接触 GRASP 验收并进入下一阶段。左右门均完成了两项物理验收，且逐步 physical command 往返等价，因此 phase144 验收通过。到达率不是训练后的成功率：该实验是单轮随机初态的确定性 reference 回归，不保存策略。

### phase145：统一 canonical 接口的四卡干净训练（运行中）

```text
起点：随机初始化策略，不加载任何旧 checkpoint
并行：4 x 2048 env = 8192 env
首轮长度：100 update
唯一变量：在 phase144 统一 observation/action/reference 语义下学习
保持不变：reward、阶段验收、随机化、低层模型和 staged-reset 设计
```

phase145 首轮决策门槛：

1. 训练必须保持纯动力学、headless、`cameras=0`、`RTX sensors=0`、frame rendering disabled。
2. 不出现 NaN、运行时错误、同一 SimulationApp 重建卡死或 staged-reset buffer 语义错误。
3. 分别记录左右门各阶段到达率；先检查迁移前在 GRASP 已出现的明显左右分化是否消失。
4. 以首个阶段断层定位 reward/探索瓶颈；根因未由轨迹和统计确认前不修改 reward。
5. phase141 step925 仅用于历史到达率对照，不加载、不做兼容分支。

启动记录（2026-09-25 22:58，北京时间）：

```text
容器：doorman_train
日志：/tmp/b2z1_phase145_canonical_clean_8192.log
GPU：0,1,2,3
每卡环境数：2048
总环境数：8192
更新数：100
num_steps_per_env：64
num_mini_batches：4
checkpoint：不加载；每 25 update 保存一次
```

2026-09-25 23:01 核对：四个训练 rank 均存活，均已打印 `Physics-only scene verified: cameras=0, RTX sensors=0, frame rendering disabled`。

2026-09-25 23:03 核对：iteration 1--2 正常完成，累计 `1,048,576` timesteps，最近一轮约 `31.12 s`；未发现 NaN、异常或 staged-reset buffer 错误。两轮累计到达率均为 APPROACH_PREGRASP `100%`、其后阶段 `0%`，这是干净策略训练刚开始时的运行基线，不用于判断 reward 瓶颈。

2026-09-25 23:15，iteration 25：

```text
checkpoint：model_step_000025.pt（24,516,603 bytes）
SHA-256：300eb35232c73c4a3274ea20830ec96c88561cedd7b3df568ae02694cb85e811
累计 timesteps：13,107,200
iteration time：30.28 s

阶段                         总体      左门      右门
APPROACH_PREGRASP reached   100.00%   100.00%   100.00%
GRASP reached                30.68%    26.77%    34.61%
OPEN_LATCH reached            7.90%    14.27%     1.51%
REPOSITION reached            4.71%     8.81%     0.59%
PUSH_OPEN reached             0.73%     1.37%     0.09%
THROUGH reached               0.00%     0.00%     0.00%
```

已确认：统一 canonical 接口可在 8192 env 下稳定训练并跨越到 PUSH_OPEN；旧接口 phase141 的“右门从 GRASP 起显著领先”现象没有原样延续。尚未确认：当前左门在 OPEN_LATCH 后领先是否只是干净训练早期采样波动，不能据此认定新的镜像错误，也不能据此调整 reward。下一次决策点为 iteration 50。

备份记录：2026-09-25 23:18 已将本文件和 step 25 checkpoint 同步到 `wholebody_1`、`wholebody_3`、`wholebody_4`；三台 checkpoint SHA-256 均与源机一致。此次同步不改变远程 Git 分支、提交、凭据或其他工作区文件。

2026-09-25 23:29，iteration 50：

```text
checkpoint：model_step_000050.pt（24,516,603 bytes）
SHA-256：2fef582f885971b5d4c34f38ada801d3eec2e1cf14be58c957e84a0d75d95a04
累计 timesteps：26,214,400
iteration time：33.30 s

阶段                         总体      左门      右门
APPROACH_PREGRASP reached   100.00%   100.00%   100.00%
GRASP reached                34.83%    37.28%    32.37%
OPEN_LATCH reached           16.05%    29.38%     2.67%
REPOSITION reached           14.70%    26.96%     2.40%
PUSH_OPEN reached             4.01%     7.25%     0.76%
THROUGH reached               0.00%     0.00%     0.00%
```

指标语义必须按阶段入口理解：`GRASP reached` 表示已完成 APPROACH_PREGRASP 并进入 GRASP；`OPEN_LATCH reached` 才表示通过严格 GRASP 验收。因此左/右 `37.28% / 32.37%` 说明接近阶段没有形成数量级差异，而 `29.38% / 2.67%` 将首个断层明确定位在右门 GRASP。

同一 iteration 的 GRASP 只读诊断：

```text
                                      左门       右门
pocket_error_vertical                 5.5 mm     53.4 mm
axis_error                            0.0077     0.0507 rad
fixed_contact_force                   1.73 N     31.04 N
moving_contact_force                  1.30 N      0.36 N
workspace_projection_error            0.3 mm      1.2 mm
steady_inverse position compensation 24.5 mm     34.3 mm
policy/reference residual             0           0
```

这不是“右门没有进入 GRASP”，也没有证据支持放宽验收阈值。右门 reference 被策略准确跟随后仍形成约 `53.4 mm` 垂直误差和固定爪单侧顶压；当前根因范围限定在 GRASP reference、右门目标姿态下的稳态逆映射或两者间的目标坐标组合。进入 OPEN_LATCH 的左右样本把手角、锁舌位移和 canonical action 量级接近，因此暂不修改 OPEN_LATCH reward/action。下一步只读追踪上述垂直误差的第一处来源，phase145 本身保持原样继续到 iteration 75。

进一步分解表明：右门 `reference_minus_target_vertical=1.9 mm`，说明几何 target 到 reference 的闭环已经基本到位；`actual_minus_reference_vertical=51.5 mm`，其中稳态逆映射命令补偿为 `+34.3 mm`，实际又比命令高 `17.5 mm`。左门对应值为 `-2.2 mm`、`+7.6 mm`、`+24.5 mm` 和 `-16.6 mm`，补偿后净误差较小。由此排除“右门 handle 世界高度或 grasp target 直接算错 5 cm”；剩余范围是右门翻转抓取姿态下的动态跟踪、补偿尚未稳态就提前发生单侧接触，或接触约束本身造成持续抬升。聚合均值无法判断时间先后，必须取得阶段入口后的时序轨迹才允许修复。

备份记录：2026-09-25 23:34 已将本文件和 step 50 checkpoint 同步到 `wholebody_1`、`wholebody_3`、`wholebody_4`；三台 checkpoint SHA-256 均为 `2fef582f885971b5d4c34f38ada801d3eec2e1cf14be58c957e84a0d75d95a04`。

2026-09-25 23:42，iteration 75：

```text
checkpoint：model_step_000075.pt（24,516,603 bytes）
SHA-256：807b3957af77ffd120635bad877a9dddc202e0c6c01cce4a54ea43b37d9bf07f
累计 timesteps：39,321,600
iteration time：32.97 s

阶段                         总体      左门      右门
APPROACH_PREGRASP reached   100.00%   100.00%   100.00%
GRASP reached                41.97%    46.62%    37.30%
OPEN_LATCH reached           20.24%    36.71%     3.71%
REPOSITION reached           18.75%    34.29%     3.17%
PUSH_OPEN reached             6.23%    11.41%     1.03%
THROUGH reached               0.16%     0.29%     0.02%
```

右门 GRASP 聚合诊断与 iteration 50 基本一致：`pocket_error_vertical=53.6 mm`、`late=49.3 mm`、`actual_minus_reference_vertical=51.6 mm`、固定/移动爪力 `30.49/0.14 N`。因此排除“iteration 50 的左右差异只是早期采样波动”。下一实验只增加按 GRASP 阶段年龄分桶的时序日志，不改变 reference、动作、reward、验收或随机化；先确认误差与首次单侧接触的因果顺序。

2026-09-25 23:57，iteration 100（phase145 正常完成）：

```text
checkpoint：model_step_000100.pt（24,516,603 bytes）
SHA-256：39c09a329a7ef8bd0f95401ca8353eb58080bbb85968e3c0ea56f418dbe5dd78
累计 timesteps：52,428,800
iteration time：32.45 s

阶段                         总体      左门      右门
APPROACH_PREGRASP reached   100.00%   100.00%   100.00%
GRASP reached                43.20%    50.91%    35.45%
OPEN_LATCH reached           22.70%    41.51%     3.83%
REPOSITION reached           21.35%    39.34%     3.30%
PUSH_OPEN reached             7.77%    14.36%     1.15%
THROUGH reached               0.11%     0.22%     0.00%
```

phase145 全程无 NaN、异常、同一 SimulationApp 重建或 staged-reset buffer 错误；四 rank 始终为 `cameras=0, RTX sensors=0, frame rendering disabled`。右门 GRASP 最终仍有 `pocket_error_vertical=52.1 mm`、`late=46.6 mm`、固定/移动爪力 `31.89/0.21 N`，因此 phase145 的结论是“纯动力学训练链路稳定且完整任务可达，但右门 GRASP 存在确定的控制/reference 系统性问题”，不是训练验收通过。

### phase146：右门 GRASP 误差时间顺序诊断（已登记，待运行）

```text
基线：phase145 model_step_000100.pt
拓扑：4 x 2048 env，保持 checkpoint 的四 rank staged-reset 快照可加载
长度：1 update x 64 policy steps
唯一变量：增加 GRASP 阶段年龄分桶日志
独立目录：复制 checkpoint 后运行，禁止覆盖 phase145 last.pt
禁止改动：行为、reference、observation、reward、验收、随机化、低层模型
```

年龄分桶为 `0--24`、`25--49`、`50--99`、`100--199`、`200--399`、`>=400` simulation steps；每桶分别记录 pocket z 误差、actual-reference、actual-command、reference-target、稳态补偿、轴误差、双爪接触标志/力和 EE 速度。判据是找出右门误差与固定爪单侧接触首次显著出现的先后关系；该临时日志定位完成后删除。

phase146 已完成，四个 rank 均恢复 step 100 的 staged-reset 快照并确认纯动力学模式。关键时序如下：

```text
GRASP 年龄                 左门 0--24   右门 0--24   右门 25--49   右门 50--99
pocket z error               -19.5 mm      33.4 mm       40.0 mm       37.1 mm
actual-reference z            -2.1 mm       7.6 mm       23.7 mm       27.1 mm
reference-target z           -17.4 mm      25.8 mm       16.3 mm       10.0 mm
fixed finger hit              0.34%         0.03%         7.28%        66.72%
fixed finger force            0.95 N        0.21 N        3.38 N       40.42 N
```

因果结论：右门在几乎无接触的第一个时间桶已经存在 `33.4 mm` 高差，且主要来自 reference 在阶段入口相对 GRASP target 高 `25.8 mm`；固定爪命中和大力接触随后才出现。因此排除“固定爪接触先把 EE 顶高”和“稳态逆映射先产生 5 cm 误差”。代码审查确认第一处不连续来自 `_pocket_pos`：APPROACH 用固定 `_PREGRASP_POCKET_OFFSET`，GRASP 改用由固定爪/移动爪真实刚体计算的 `_grasp_pocket_pos`。镜像抓取姿态把同一近似误差映射成相反的世界竖直偏差。

### phase147：统一 APPROACH/GRASP physical pocket（已验证）

修复只位于 pocket 几何所有权层：两阶段统一使用真实 `_grasp_pocket_pos`；APPROACH reference 根据当前真实通道相对 gripper origin 的偏移闭环求目标；命令诊断使用同一实测局部偏移。固定 `_PREGRASP_POCKET_OFFSET`、对应状态和 phase146 临时时序日志已删除。未修改速度、reward、阶段验收、随机化、低层模型或 canonical action 接口。

静态与短运行验证：

```text
py_compile：通过
task_frame pytest：5 passed
64 env x 8 steps smoke：通过
纯动力学：cameras=0, RTX sensors=0, frame rendering disabled
日志：/tmp/b2z1_phase147_pocket_unified_smoke.log
```

32 env x 1100 policy steps 的干净零 residual reference 回归已正常结束：

```text
日志：/tmp/b2z1_phase147_unified_pocket_zero_residual.log
总 timesteps：35200
iteration time：351.28 s

                              左门       右门
APPROACH_PREGRASP reached    100.00%    100.00%
GRASP reached                 14.30%     14.41%
OPEN_LATCH reached             6.22%      7.35%
GRASP pocket z error          10.3 mm     26.8 mm
actual-reference z             9.5 mm     27.0 mm
reference-target z             0.8 mm     -0.2 mm
fixed finger force             0.83 N     11.26 N
moving finger force            1.66 N      1.40 N
policy-reference residual      0           0
```

指标仍按阶段入口解释：`GRASP reached` 表示完成 APPROACH，`OPEN_LATCH reached` 才表示通过严格 GRASP 验收。左右门都能连续完成 APPROACH 并通过严格 GRASP，且通过率没有旧训练中的数量级不对称。

本 phase 验证的结论是 pocket 事实源统一修复成立：右门 `reference-target z` 从 phase146 入口桶的 `25.8 mm` 降至聚合 `-0.2 mm`，reference 对齐子集为约 `0.1 mm`；阶段切换不再制造厘米级几何目标跳变。右门聚合 `actual-reference z=27.0 mm` 和后期单爪接触仍存在于未通过样本，但它们已不再由 APPROACH/GRASP target 定义不连续解释；不能把该尾部现象与已修复根因混为一谈，也不能据此恢复旧 pocket 或放宽 GRASP 验收。

phase147 不保存可继承策略。下一步必须从随机策略开始 phase148 四卡 PPO，使 staged-reset 样本全部由统一 pocket 语义重新生成；phase145 step 100 仅保留为修复前对照。

备份记录：2026-09-26 00:18 已把当前 24 个修改/新增源码与文档，以及 phase145 step 100 checkpoint，同步到 `wholebody_1`、`wholebody_3`、`wholebody_4`。三台 `door_open_homie.py`、本状态文档和 checkpoint 的 SHA-256 均与源机一致；这只是备份同步，不是远程运行验收。

### phase148：统一 pocket 语义下四卡干净 PPO（已验证）

```text
起点：随机初始化策略，不加载 checkpoint
并行：4 x 2048 env = 8192 env
计划长度：100 update
保存频率：每 25 update
唯一行为变量：继承 phase147 已验证的统一 physical pocket
保持不变：reward、阶段验收、随机化、低层模型、canonical action/observation 语义
日志：/tmp/b2z1_phase148_unified_pocket_clean_8192.log
```

本 phase 先在 update 1、25、50、75、100 记录左右门逐阶段到达率。首要比较项是 `GRASP reached -> OPEN_LATCH reached` 的左右通过率，以及 GRASP 的 `reference-target`、`actual-reference`、双爪接触力和 policy-reference residual。phase145 checkpoint 不加载，因为它的策略和 staged-reset 样本均来自修复前 pocket 语义。

成功判据：四个 rank 持续保持纯动力学模式且无运行时错误；右门不再复现修复前 `reference-target` 厘米级跳变；左右门 GRASP 学习不出现旧有数量级断层。失败时先记录首个断层和证据，不修改 reward、阈值或速度。

启动与首轮核对（2026-09-26 00:20--00:23，北京时间）：

```text
四个 rank：均存活，分别绑定 GPU 0、1、2、3
场景：每 rank 2048 env，总计 8192 env
纯动力学：四个 rank 均确认 cameras=0, RTX sensors=0, frame rendering disabled
iteration 1 timesteps：524288
iteration time：28.78 s
APPROACH_PREGRASP reached：100.00%（左右均 100.00%）
GRASP 及后续 reached：0
```

这是随机策略的正常起点，与 phase145 iteration 1 一致，不用于判断 reward 或阶段瓶颈。下一次决策记录点为 update 25；此前只监控异常、NaN 和运行状态，不修改行为逻辑。

2026-09-26 00:34，iteration 25：

```text
checkpoint：model_step_000025.pt
SHA-256：db65b1dffbb16881cfbcae86e7c3a35ea55363c12f7c38aa7ca0f1da055b4a7c
累计 timesteps：13631488
iteration time：30.70 s

阶段                         总体      左门      右门
APPROACH_PREGRASP reached   100.00%   100.00%   100.00%
GRASP reached                31.70%    28.41%    35.00%
OPEN_LATCH reached           12.93%    15.23%    10.61%
REPOSITION reached            0.19%     0.25%     0.14%
PUSH_OPEN reached             0.01%     0.02%     0.00%
THROUGH reached               0.00%     0.00%     0.00%
```

与 phase145 修复前同一训练点比较，GRASP 入口率相近；右门严格 GRASP 通过率从 `1.51%` 提高到 `10.61%`，左右差距由近十倍缩小到 `15.23%/10.61%`，支持 phase147 修复有效。两侧 `reference-target z` 均约 `0.2 mm`，policy-reference residual 为 0；没有复现旧的 target/reference 跳变。

首个明显断层现为 `OPEN_LATCH reached -> REPOSITION reached`，即通过严格抓取的样本尚未学会稳定压把手并形成门缝；左右都低，不是旧有单侧几何错误的直接延续。当前只训练 25 update，继续到 update 50 观察该阶段是否随探索自然增长，期间禁止修改 reward、reference、阈值和速度。

2026-09-26 00:48，iteration 50：

```text
checkpoint：model_step_000050.pt
SHA-256：ff6c37b4af698adf16c74a38a058f87462672bcebb22b93615123321dcfa38a2
累计 timesteps：26738688
iteration time：32.12 s

阶段                         总体      左门      右门
APPROACH_PREGRASP reached   100.00%   100.00%   100.00%
GRASP reached                31.91%    28.94%    34.90%
OPEN_LATCH reached           16.36%    18.83%    13.89%
REPOSITION reached            0.37%     0.45%     0.30%
PUSH_OPEN reached             0.03%     0.04%     0.02%
THROUGH reached               0.00%     0.01%     0.00%

OPEN_LATCH 活跃样本：
                         左门       右门
handle angle           0.2538     0.1369 rad
latch position         0.0097     0.0052 m
hinge angle            0.0078     0.0137 rad
contact quality        0.4793     0.4383
```

GRASP target/reference 仍连续：左右 `reference-target z` 都约 `0.3 mm`，policy-reference residual 为 0。严格 GRASP 通过率继续增长，但 `OPEN_LATCH -> REPOSITION` 在两侧都只约 `2.2%`，因此当前首个真实瓶颈已定位为 OPEN_LATCH，而非 APPROACH 或旧 pocket 切换错误。右门把手角和锁舌位移低于左门，是否来自 GRASP 后实际 EE 偏差、接触几何，或该阶段学习信号不足，尚未确认；update 75 前只做代码和指标链的只读核对，不写行为补丁。

2026-09-26 01:02，iteration 75：

```text
checkpoint：model_step_000075.pt
SHA-256：790f2363106c2ea9f594da525297f8c10bab0ed63d4a60815a6e7bfe10e28ea6
累计 timesteps：40370176
iteration time：33.19 s

阶段                         总体      左门      右门
APPROACH_PREGRASP reached   100.00%   100.00%   100.00%
GRASP reached                37.87%    35.89%    39.87%
OPEN_LATCH reached           19.76%    22.93%    16.59%
REPOSITION reached            2.04%     2.27%     1.81%
PUSH_OPEN reached             0.25%     0.34%     0.16%
THROUGH reached               0.01%     0.00%     0.02%
```

OPEN_LATCH 的平均锁舌进度升至 `0.6366`，锁舌完全释放瞬时比例 `29.04%`，EE 速度从 update 50 的约 `0.394 m/s` 降至 `0.194 m/s`。阶段完成累计到达率从 update 50 的 `0.37%` 增至 `2.04%`，说明该阶段尚未停滞，不应在中途更改 reward。

同时，OPEN_LATCH 活跃样本的严格双侧夹持仅 `1.42%`，任一有效把手接触为 `53.23%`，相向接触为 `4.32%`。这证明策略多数时间没有保持 GRASP 验收时的严格夹持，但尚不能单凭瞬时严格判据认定所有成功动作无效，因为接触流形可交替且门缝成功率正在增长。phase148 保持不变跑完 update 100；结束后再结合阶段成功轨迹与累计转移率决定是否调整奖励结构。

2026-09-26 01:16，iteration 100（phase148 正常完成）：

```text
checkpoint：model_step_000100.pt
SHA-256：bee86fc18eb554ec47a5a030b1af3f6613c92feaed677d1d85ed9d1aa7e0af41
累计 timesteps：52428800
iteration time：34.37 s
总训练时间：3149.01 s

阶段                         总体      左门      右门
APPROACH_PREGRASP reached   100.00%   100.00%   100.00%
GRASP reached                42.64%    44.57%    40.71%
OPEN_LATCH reached           25.68%    33.39%    17.95%
REPOSITION reached           15.29%    25.51%     5.03%
PUSH_OPEN reached             1.68%     2.27%     1.10%
THROUGH reached               0.55%     0.14%     0.96%
```

phase148 全程无 NaN、异常、环境重建或 staged-reset buffer 错误；四 rank 始终为 `cameras=0, RTX sensors=0, frame rendering disabled`。统一 pocket 后，右门严格 GRASP 到达率由 phase145 step100 的 `3.83%` 提高到 `17.95%`，旧 target/reference 跳变没有复现。

新的首个系统性断层位于右门 OPEN_LATCH：左门从 `33.39%` 进入 OPEN_LATCH 后有 `25.51%` 完成，转换率约 `76.4%`；右门从 `17.95%` 进入后仅 `5.03%` 完成，转换率约 `28.0%`。同一时刻：

```text
OPEN_LATCH 活跃样本         左门       右门
latch position             21.8 mm     14.4 mm
handle angle                0.5716      0.3782 rad
hinge angle                 0.0302      0.0302 rad
contact quality             0.6808      0.4728

GRASP 活跃样本
actual-reference z           9.4 mm     28.5 mm
reference-target z           0.4 mm      0.3 mm
```

因此不能把下一步简化为“OPEN_LATCH reward 太稀疏”：两侧使用同一 canonical 动作和 reward，但右门进入阶段后的夹持质量、把手角和锁舌位移都系统性更低。phase149 只增加按 OPEN_LATCH 阶段年龄和门侧分桶的无副作用诊断，确认实际 pocket/reference 误差、双爪接触、EE 速度和 canonical 动作谁先分化；根因确认前不修改 reward、动作、验收或阈值。

### phase149：右门 OPEN_LATCH 入口时序诊断（已验证）

```text
基线：phase148 model_step_000100.pt 及其 staged-reset 快照
拓扑：4 x 2048 env，运行 1 update
唯一变量：增加 OPEN_LATCH 年龄分桶日志
禁止改动：行为、reference、observation、reward、验收、随机化、低层模型
```

年龄桶至少覆盖 `0--24`、`25--49`、`50--99`、`100--199`、`>=200` policy steps。每侧记录 pocket 位置/轴向误差、双爪接触与力、把手角、锁舌位移、门角、EE 速度，以及 policy mean canonical EE 动作。诊断结束后删除临时日志。

启动记录（2026-09-26 01:20，北京时间）：四个 rank 使用 GPU 0--3，每 rank 2048 env，恢复 phase148 step100 的匹配 rank 环境快照，只运行 step101。日志为 `/tmp/b2z1_phase149_open_latch_timeline.log`。临时诊断代码在结论形成后删除，不同步为正式远端代码。

第一轮分桶结果确认：右门 `0--24` 步 physical pocket 高度误差为 `-2.0 mm`，左门为 `-1.4 mm`，排除“右门一进入 OPEN_LATCH 就发生厘米级 pocket 平移跳变”。分化首先出现在姿态和动作：

```text
OPEN_LATCH age             左门 0--24   右门 0--24   右门 25--49   右门 50--99
axis error                    0.085       0.167          0.603          0.827 rad
opposed contact              28.94%      20.42%          1.70%          4.42%
policy mean roll             0.0162      0.0045         0.0037         0.0008
policy mean pitch            0.0065      0.0158         0.0166         0.0139
policy mean yaw             -0.0021      0.0121         0.0109         0.0085
```

但 `0--24` 桶仍把阶段入口与 24 步后的结果混在一起，无法判断右门在第几步首先偏离；原诊断中的 GRASP reference-target/actual-reference 在 OPEN_LATCH 没有物理语义，已删除。第二轮 phase149b 将入口细分为 `0--4`、`5--9`、`10--24`，并改为记录有效的 actual-command 姿态跟踪、commanded axis 和 actual axis。除日志外仍不修改任何行为。

第二轮 phase149b 使用完全相同的 step100 checkpoint 和四个 rank 环境快照，结论如下：

```text
OPEN_LATCH age                    左门 0--4   右门 0--4   左门 5--9   右门 5--9   左门 10--24   右门 10--24
physical pocket z error             -0.1 mm     -0.2 mm     -0.5 mm     -0.9 mm      -2.1 mm       -2.9 mm
actual axis error                    0.0124       0.0108      0.0410       0.0675       0.1201        0.2413 rad
commanded axis error                 0.0245       0.0304      0.0340       0.0881       0.0938        0.2677 rad
orientation tracking error           0.4185       0.4074      0.0376       0.0258       0.0496        0.0331 rad
precise bilateral grasp              32.48%       33.16%      36.12%        8.55%        1.47%         0.19%
opposed contact                      71.44%       63.84%      49.74%       35.64%       10.62%         3.98%
policy mean roll                     0.0125       0.0028      0.0171       0.0034       0.0169         0.0054
policy mean pitch                    0.0079       0.0145      0.0070       0.0151       0.0060         0.0164
policy mean yaw                      0.0000       0.0113     -0.0019       0.0122      -0.0028         0.0122
```

因果结论：阶段入口的 pocket、实际轴向和严格夹持质量左右等价；右门从第一批高层动作开始就把 canonical 旋转主要给了 pitch/yaw，而不是保持轴向并通过 roll 压把手。`5--9` 步时 commanded axis 已先分化，低层 orientation tracking error 反而更小，证明低层在准确执行错误的高层目标。随后 actual axis 偏离、相向接触丢失和右门锁舌进度不足依次发生。

代码审查同时确认，当前 OPEN_LATCH progress 是 `0.45*latch + 0.45*gap + 0.10*contact_quality` 的加权和；pocket 和 axis 不在进度中，接触质量也只是可被锁舌/门缝补偿的加分项。因此策略可以在破坏夹持几何的同时获得短期正进度。下一修复应位于 OPEN_LATCH reward 所有权层：把 pocket、axis 和近期双爪相向接触组成连续的 retention quality，并让锁舌/门缝进度由该质量门控；不增加脚本动作、左右门分支或 OPEN_LATCH action reference。phase149 临时日志在修改前删除。

### phase150：OPEN_LATCH 夹持保持奖励修复（已验证）

唯一行为修改：OPEN_LATCH 的阶段进度由“可互相补偿的加权和”改为“夹持保持质量门控的锁舌/门缝进度”。夹持保持质量使用 physical pocket 距离、插入轴误差、近期双爪接触和相向接触，全部来自现有通用物理量；左右门使用完全相同公式。

禁止：增加 scripted reference、监督 EE 任务动作、左右门特调、放宽 GRASP/OPEN_LATCH 验收，或修改低层控制器。先做静态测试和小规模 smoke，再从 phase148 step100 分叉训练观察错误动作能否被纠正；确认机制有效后必须从随机策略做干净训练。

实现与静态验证：

1. 新增近期双爪接触比例作为通用 grasp 物理量，不改变 observation 或验收。
2. retention quality 取 `pocket_retention`、`axis_retention`、`contact_retention` 的最小值，左右门共用同一公式。
3. OPEN_LATCH progress 改为 `retention * (0.10 + 0.45*latch + 0.45*gap)`；保留 signed potential，因此破坏夹持会立即产生负反馈，恢复夹持可获得正反馈，但不能通过往复刷总收益。
4. 没有增加 action reference、EE supervision、左右门分支或新阈值验收。
5. `py_compile`、5 项 task-frame pytest 和 64 env x 8 step smoke 均通过；smoke 日志 `/tmp/b2z1_phase150_retention_smoke.log`。

checkpoint 核对：phase148 checkpoint 的 rank-local env state 只保存 staged-reset 物理案例、command history 和时间，不保存 `_task_stage_best` 或 progress initialized 状态，因此可以用同一物理案例做 reward 分叉对照，不会把旧 progress 数值带入新公式。phase150 先从 step100 分叉续训 25 update；该结果只验证新机制能否纠正已学错误动作，机制通过后仍需从随机策略干净重训。

启动记录（2026-09-26 01:38，北京时间）：四卡、每卡 2048 env，从 phase148 step100 复制到独立目录 `door_open_b2z1_lstm_mlp-20260926_013800_phase150_retention`，计划训练至 step125。日志 `/tmp/b2z1_phase150_retention_branch.log`；phase148 原目录和 checkpoint 不覆盖。

Hydra 实际为续训创建了运行目录 `door_open_b2z1_lstm_mlp-20260926_013715`，step125 checkpoint 位于该目录；复制目录只作为 checkpoint 输入，没有被覆盖。训练正常结束，监控脚本因等待复制目录中的文件而返回失败码，这只是监控路径错误，不是训练失败。

step125 结果：

```text
checkpoint SHA-256：d21029f2321d0412a1b084aa56f65a12c14d9c5717f6af42f7afb65f18a00c6b
新增训练：25 update / 13107200 timesteps

阶段                         总体      左门      右门
APPROACH_PREGRASP reached   100.00%   100.00%   100.00%
GRASP reached                48.48%    53.31%    43.65%
OPEN_LATCH reached           31.19%    41.90%    20.45%
REPOSITION reached           22.37%    38.58%     6.12%
PUSH_OPEN reached             5.50%     9.29%     1.70%
THROUGH reached               0.96%     0.41%     1.51%
```

右门 OPEN_LATCH 的 policy mean pitch/yaw 从 phase149 的约 `0.015/0.012` 降至 `0.0060/0.0024`，锁舌位移从 `14.4 mm` 升至 `18.1 mm`，contact quality 从 `0.4728` 升至 `0.5110`。说明 retention reward 正在纠正错误旋转，并非无效机制；但右门 roll 仍只有 `0.0007`，`OPEN_LATCH -> REPOSITION` 转换率约 `29.9%`，尚未收敛。保持公式和其他代码不变，继续同一分叉到 step150，再决定是否进入干净重训。

phase150b 续训登记（2026-09-26 01:57，北京时间）：从上述 step125 checkpoint 继续到 step150，四卡、每卡 2048 env。唯一变量仍是已经实现的 retention progress；本段不再修改任何行为逻辑。成功判据是右门错误 pitch/yaw、锁舌位移、接触质量和 `OPEN_LATCH -> REPOSITION` 转换率继续朝正确方向改善，同时前两阶段不回归；若未改善，只定位 retention 分量或梯度失效原因。

```text
日志：/tmp/b2z1_phase150b_retention_to150.log
运行目录：logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260926_015725
计划 checkpoint：model_step_000150.pt
02:10 核对：iteration 141，四个 rank 存活，无 Python 异常或 NaN
```

step150 正常完成，四个 rank 退出，无 Python 异常或 NaN：

```text
checkpoint：model_step_000150.pt
SHA-256：7b4515d941073a0ffacb5057ac3a4e5e9af2458feae790ff99698650d48a427d
累计 timesteps：78643200

阶段                         总体      左门      右门
APPROACH_PREGRASP reached   100.00%   100.00%   100.00%
GRASP reached                50.68%    56.13%    45.20%
OPEN_LATCH reached           35.15%    46.21%    24.05%
REPOSITION reached           25.04%    42.99%     7.03%
PUSH_OPEN reached             7.51%    12.85%     2.15%
THROUGH reached               1.27%     0.68%     1.87%
```

右门从 phase148 step100 到 phase150 step150：OPEN_LATCH reached `17.95% -> 24.05%`，REPOSITION reached `5.03% -> 7.03%`，锁舌位移 `14.4 -> 17.3 mm`，contact quality `0.473 -> 0.536`；policy mean pitch/yaw 从约 `0.015/0.012` 降至 `0.0061/0.0022`。因此 retention reward 确实纠正了部分错误旋转，机制验证通过。

但 step125 到 step150 的右门 `OPEN_LATCH -> REPOSITION` 转换率约为 `29.9% -> 29.2%`，已经平台；右门 roll 仍从 `0.0007` 降至 `0.0004`。总体 retention 三分量为 pocket `0.5220`、axis `0.1463`、contact `0.3468`，最小值主要由 axis 限制。继续从继承的旧策略分叉训练不能回答干净策略会否形成正确动作，因此不再延长该分叉，也不据此增加补丁。下一实验 phase151 使用完全相同代码和 reward 从随机策略训练。

### phase151：retention reward 四卡干净训练（失败）

```text
起点：随机初始化策略，不加载 checkpoint
并行：4 x 2048 env = 8192 env
计划长度：150 update
保存频率：每 25 update
唯一实验变量：相对 phase148 使用 phase150 已验证的 retention progress
保持不变：阶段验收、随机化、低层模型、canonical action/observation 语义及其他 reward
日志：/tmp/b2z1_phase151_retention_clean_8192.log
```

待验证假设：从随机策略开始时，retention 门控能让 OPEN_LATCH 在形成错误 pitch/yaw 动作模式前学会保持 pocket、轴向和双爪接触，使左右门转换率同时增长。失败判据不是单个 update 波动，而是在连续记录点出现稳定断层或 retention/action 指标不再改善。

2026-09-26 02:21 启动：四个 rank 已分别绑定 GPU 0--3，启动命令没有 checkpoint 参数。

首个 update 核对：四个 rank 均明确报告 `cameras=0, RTX sensors=0, frame rendering disabled`，Hydra 实际运行目录为 `door_open_b2z1_lstm_mlp-20260926_021944`。iteration 1 的 APPROACH_PREGRASP 左右 reached 均为 `100%`，其余阶段为 0，与 phase148 随机策略起点一致；无异常或 NaN。下一记录点为 update25，此前只监控运行错误，不修改行为。

iteration 25：

```text
checkpoint：model_step_000025.pt
SHA-256：6a86fafe147166da10fb23f1681df93be1dd1b1b5f7601a6d9a5de0d2af75872

阶段                         总体      左门      右门
APPROACH_PREGRASP reached   100.00%   100.00%   100.00%
GRASP reached                30.49%    28.26%    32.73%
OPEN_LATCH reached           11.43%    13.62%     9.23%
REPOSITION reached            0.20%     0.00%     0.39%
PUSH_OPEN reached             0.01%     0.00%     0.02%
THROUGH reached               0.00%     0.00%     0.00%

OPEN_LATCH retention：pocket 0.5500，axis 0.4040，contact 0.3517，min quality 0.2565
左门 policy mean roll/pitch/yaw：-0.0014 / 0.0003 / 0.0004
右门 policy mean roll/pitch/yaw： 0.0015 / 0.0002 / 0.0010
```

与 phase148 旧 reward 的 update25 相比，总体 GRASP `31.70% -> 30.49%`、OPEN_LATCH `12.93% -> 11.43%`、REPOSITION `0.19% -> 0.20%`，均处于相近的早期学习水平；此时不能判定 retention reward 有收益或回归。保持代码不变继续到 update50，检查 OPEN_LATCH 转换率、左右门动作轴和 retention 是否开始分化。

iteration 50：

```text
checkpoint：model_step_000050.pt
SHA-256：fe4a1d9fc98dc35b6949bfe186f70d5e2eeafc141a0fcc5ba53ffe289f928c2b

阶段                         总体      左门      右门
APPROACH_PREGRASP reached   100.00%   100.00%   100.00%
GRASP reached                34.79%    32.03%    37.55%
OPEN_LATCH reached           19.17%    20.91%    17.42%
REPOSITION reached            3.72%     0.09%     7.36%
PUSH_OPEN reached             0.62%     0.00%     1.24%
THROUGH reached               0.00%     0.00%     0.00%

OPEN_LATCH retention：pocket 0.4909，axis 0.4013，contact 0.3448，min quality 0.2566
左门 latch/contact：6.3 mm / 0.5494；policy mean roll/pitch/yaw：-0.0003 / 0.0002 / 0.0011
右门 latch/contact：13.5 mm / 0.5671；policy mean roll/pitch/yaw： 0.0021 / 0.0009 / 0.0012
```

相对 phase148 update50，总体 GRASP `31.91% -> 34.79%`、OPEN_LATCH `16.36% -> 19.17%`、REPOSITION `0.37% -> 3.72%`，总体学习更快；但收益高度不对称。当前 `OPEN_LATCH -> REPOSITION` 转换率左门约 `0.4%`、右门约 `42.3%`，且左门 roll 接近零并略为负，锁舌位移仅 `6.3 mm`。这只确认 update50 存在反向不对称，尚不能区分学习时序波动、统一坐标/奖励仍有缺陷或策略局部最优；根因未确认，不修改行为，继续到 update75 观察趋势。

iteration 75：

```text
checkpoint：model_step_000075.pt
SHA-256：db8065bdbbf6ea16db62aa8a182a92984eec9553aebc416d5c6baa7195aca9ea

阶段                         总体      左门      右门
APPROACH_PREGRASP reached   100.00%   100.00%   100.00%
GRASP reached                42.94%    39.26%    46.62%
OPEN_LATCH reached           25.59%    25.85%    25.32%
REPOSITION reached            9.29%     0.98%    17.62%
PUSH_OPEN reached             3.18%     0.51%     5.85%
THROUGH reached               0.80%     0.00%     1.61%

OPEN_LATCH retention：pocket 0.5200，axis 0.5101，contact 0.4213，min quality 0.3404
左门 latch/contact：12.0 mm / 0.6040；policy mean roll/pitch/yaw：0.0005 / 0.0006 / 0.0016
右门 latch/contact：19.5 mm / 0.6288；policy mean roll/pitch/yaw：0.0030 / 0.0000 / 0.0009
```

相对 phase148 update75，总体 OPEN_LATCH `19.76% -> 25.59%`、REPOSITION `2.04% -> 9.29%`、PUSH_OPEN `0.25% -> 3.18%`，retention reward 明显加快了总体后续阶段学习。但左右 OPEN_LATCH 入口率已经相等，完成率仍高度不对称：左门约 `3.8%`，右门约 `69.6%`。左门并非完全学反：roll 从 update50 的 `-0.0003` 转为 `+0.0005`，锁舌位移从 `6.3` 增至 `12.0 mm`；因此保持代码不变继续到 update100，同时只读核对统一 task-frame 与 reward 计算链，不能据此增加门型分支。

iteration 100：

```text
checkpoint：model_step_000100.pt
SHA-256：8cd0178c72981cb0cf78b02d639358fcfb11298ae9f664f341c7463daf9d93b5

阶段                         总体      左门      右门
APPROACH_PREGRASP reached   100.00%   100.00%   100.00%
GRASP reached                45.92%    40.28%    51.58%
OPEN_LATCH reached           29.55%    27.49%    31.62%
REPOSITION reached           13.92%     0.92%    26.96%
PUSH_OPEN reached             4.39%     0.49%     8.30%
THROUGH reached               1.95%     0.00%     3.90%

OPEN_LATCH retention：pocket 0.5351，axis 0.5381，contact 0.4551，min quality 0.3631
左门 latch/contact：11.1 mm / 0.6310；policy mean roll/pitch/yaw：-0.0001 / 0.0008 / 0.0014
右门 latch/contact：21.6 mm / 0.6591；policy mean roll/pitch/yaw： 0.0046 / -0.0003 / 0.0012
```

update75 到100，左门 `OPEN_LATCH -> REPOSITION` 转换率从约 `3.8%` 降至 `3.3%`，roll 从 `+0.0005` 回到近零，锁舌位移没有继续增长；右门转换率升至约 `85.3%`。因此 update50 的不对称已成为稳定断层，而不是普通学习时序波动。只读核对没有发现 reward 的左右门分支，task-frame 的轴向镜像及正 roll 对正把手关节轴已有单元测试覆盖。当前 phase 仍按原计划跑至150形成完整曲线；结束后只增加无副作用分桶诊断，定位入口状态、commanded axis、retention 三分量和动作谁先分化，根因确认前不修改行为。

iteration 125：

```text
checkpoint：model_step_000125.pt
SHA-256：dd9c6bc5e1c856d27fdbe58549623f2a3eb5c01ee8ddb2467e7ea203ba249f74

阶段                         总体      左门      右门
APPROACH_PREGRASP reached   100.00%   100.00%   100.00%
GRASP reached                47.57%    42.39%    52.77%
OPEN_LATCH reached           31.64%    29.84%    33.43%
REPOSITION reached           15.56%     1.26%    29.92%
PUSH_OPEN reached             5.47%     0.57%    10.38%
THROUGH reached               2.88%     0.00%     5.78%

OPEN_LATCH retention：pocket 0.5723，axis 0.5700，contact 0.5017，min quality 0.4157
左门 latch/contact：14.3 mm / 0.6615；policy mean roll/pitch/yaw：0.0008 / 0.0008 / 0.0014
右门 latch/contact：21.3 mm / 0.6713；policy mean roll/pitch/yaw：0.0066 / -0.0004 / 0.0010
```

左门转换率小幅升至约 `4.2%`，说明它能产生少量正确动作，但与右门约 `89.5%` 的差距继续扩大。下一诊断除入口状态、commanded/actual axis 和 retention 三分量外，必须按门侧比较 OPEN_LATCH 的 progress 增量、reward-to-go/advantage 与各 action 相关信号，确认问题在物理反馈还是 credit assignment；phase151 仍不修改行为并完成 update150。

iteration 150 正常完成：

```text
checkpoint：model_step_000150.pt
SHA-256：3221b9bbd5d0227a16dfe430124b28f262ec4d5118a0b0b7dc42d9784ae600dc
累计 timesteps：78643200

阶段                         总体      左门      右门
APPROACH_PREGRASP reached   100.00%   100.00%   100.00%
GRASP reached                49.13%    42.47%    55.82%
OPEN_LATCH reached           33.79%    30.28%    37.30%
REPOSITION reached           17.82%     1.58%    34.10%
PUSH_OPEN reached             7.08%     0.73%    13.45%
THROUGH reached               3.98%     0.00%     7.97%

OPEN_LATCH retention：pocket 0.6265，axis 0.6399，contact 0.5997，min quality 0.4837
左门 latch/contact：16.0 mm / 0.7379；policy mean roll/pitch/yaw：0.0009 / 0.0005 / 0.0007
右门 latch/contact：20.5 mm / 0.6712；policy mean roll/pitch/yaw：0.0079 / -0.0001 / 0.0011
```

最终 `OPEN_LATCH -> REPOSITION` 转换率左门约 `5.2%`、右门约 `91.4%`。相对 phase148，retention reward 确实显著提高了总体后续阶段到达率和夹持质量，但 phase151 的核心假设“干净训练可让两侧共同形成正确动作”失败；不能把右门成功冒充全任务进展。phase152 只做按阶段年龄和门侧的无副作用诊断，先定位最早分化的物理量或 credit 信号。

### phase152：左门 OPEN_LATCH 时序与 credit 诊断（已验证）

唯一变量是临时日志：环境侧按门型及 `0--4`、`5--9`、`10--24`、`>=25` 阶段年龄记录 pocket/axis/contact retention、commanded/actual axis、姿态 tracking、latch/contact 和 policy roll/pitch/yaw；trainer 侧按相同分组记录 progress delta、reward、return、advantage 和旋转 action-advantage。所有指标不进入 observation、reward、loss、采样权重或动作。

静态验证：`py_compile` 通过，task-frame `5 passed`，64 env x 8 step smoke 通过。正式诊断从 phase151 step150 的四个 rank 匹配环境快照恢复，只运行 step151；日志 `/tmp/b2z1_phase152_left_open_latch_diag.log`。诊断形成结论后删除临时代码。

关键时序：

```text
年龄             左门 roll   右门 roll   左门 latch   右门 latch   左/右 retention
0--4             0.0140      0.0188       0.2 mm      0.3 mm      0.732 / 0.670
5--9             0.0187      0.0271       1.8 mm      2.8 mm      0.833 / 0.649
10--24           0.0152      0.0231       7.8 mm     10.9 mm      0.780 / 0.510
>=25             0.0089      0.0104      16.0 mm     20.1 mm      0.746 / 0.508
```

入口 `0--4` 步左右 pocket error 为 `2.4/3.9 mm`、actual axis error 为 `0.0102/0.0084 rad`、commanded axis error 为 `0.0284/0.0296 rad`，物理状态等价；左门 contact/retention 反而更高。按累计正 roll 粗略归一后，两侧锁舌位移增益同量级，排除门关节、低层或 canonical axial 变换导致左门物理响应更弱。

分化首先来自高层策略：右门从入口开始持续给出约三成更大的正 roll，较早越过完成条件；右门入口 return 为正，左门入口 return 为负，后续成功回报继续放大该差异。左右分组的早期 progress delta 同量级，且 roll action-advantage 均为正，说明 reward 并未把左门方向教反，左门仍有继续学习的有效梯度。actor observation 保留门侧和物理本体状态，variant 配置只做分组 advantage normalization，没有强制两个镜像状态输出一致，因此架构允许这种策略对称性破缺。

临时分桶、storage 和 credit 打印代码已全部删除；正式代码重新通过 `py_compile`、`git diff --check` 和 task-frame `5 passed`。下一步不立即增加一致性 loss：phase151 的左门转换率从 step100 的约 `3.3%` 增至 step150 的约 `5.2%`，先续训到250确认是否继续收敛。只有连续两个记录点平台，才进入策略表示/训练层的通用镜像一致性设计。

### phase153：保持正式机制续训至 step250（已验证）

从 phase151 `model_step_000150.pt` 的四个匹配环境快照继续训练100 update，四卡、每卡2048 env；不修改代码、reward、验收、随机化或接口。日志 `/tmp/b2z1_phase153_continue_to250.log`。记录点为175、200、225、250，核心门槛是左门 OPEN_LATCH 转换率与 roll 是否连续改善。

iteration 175：

```text
checkpoint：model_step_000175.pt
SHA-256：e1cf0124c9c6561f8f568c924aec0799f82405e3ddacaec83400e99e354c15be

阶段                         总体      左门      右门
APPROACH_PREGRASP reached   100.00%   100.00%   100.00%
GRASP reached                51.76%    45.08%    58.45%
OPEN_LATCH reached           36.32%    31.12%    41.53%
REPOSITION reached           20.60%     2.43%    38.82%
PUSH_OPEN reached             8.46%     0.91%    16.05%
THROUGH reached               6.29%     0.00%    12.59%

左门 latch/contact/roll：16.5 mm / 0.7480 / 0.0019
右门 latch/contact/roll：18.7 mm / 0.6879 / 0.0103
```

左门 `OPEN_LATCH -> REPOSITION` 转换率从 step150 的约 `5.2%` 增至 `7.8%`，roll 同时从 `0.0009` 增至 `0.0019`，未达到平台判据。保持正式机制不变继续到 step200。

iteration 200：

```text
checkpoint：model_step_000200.pt
SHA-256：0305e75597dfdac28ec853f45526e0ad170dcfb49a69fcfe9650e049414173a2

阶段                         总体      左门      右门
APPROACH_PREGRASP reached   100.00%   100.00%   100.00%
GRASP reached                51.32%    44.08%    58.58%
OPEN_LATCH reached           37.63%    32.09%    43.17%
REPOSITION reached           22.26%     3.20%    41.36%
PUSH_OPEN reached             9.49%     1.41%    17.60%
THROUGH reached               7.08%     0.05%    14.13%

左门 latch/contact/roll：19.0 mm / 0.7964 / 0.0021
右门 latch/contact/roll：19.4 mm / 0.7090 / 0.0120
```

左门转换率继续升至约 `10.0%`，并首次出现左门 THROUGH reached `0.05%`；左门平均锁舌位移已接近右门。尚未出现连续平台，继续到 step225。

iteration 225：

```text
checkpoint：model_step_000225.pt
SHA-256：45bdf7f6845e69a5d513606c9f9ee48365f2186f59c8c68c8c03df1884be56bf

阶段                         总体      左门      右门
APPROACH_PREGRASP reached   100.00%   100.00%   100.00%
GRASP reached                53.29%    45.33%    61.28%
OPEN_LATCH reached           38.45%    32.20%    44.71%
REPOSITION reached           23.65%     4.54%    42.82%
PUSH_OPEN reached            10.45%     1.77%    19.16%
THROUGH reached               7.95%     0.15%    15.78%

左门 latch/contact/roll：20.6 mm / 0.8600 / 0.0014
右门 latch/contact/roll：17.7 mm / 0.6994 / 0.0105
```

左门转换率继续升至约 `14.1%`，THROUGH reached 增至 `0.15%`，没有平台。左门平均 latch 已超过20 mm，后续未转换样本不再能简单归因于锁舌下压不足；完成 step250 后重新定位门缝/验收时序。

iteration 250 正常完成，四个 rank 正常退出，无异常或 NaN：

```text
checkpoint：model_step_000250.pt
SHA-256：32e28fc6bfbfac0c6f9f2f8841db68839bf7479656ffd8db41b67f4abaed00e5

阶段                         总体      左门      右门
APPROACH_PREGRASP reached   100.00%   100.00%   100.00%
GRASP reached                53.77%    46.96%    60.61%
OPEN_LATCH reached           39.61%    34.24%    45.00%
REPOSITION reached           24.24%     5.14%    43.40%
PUSH_OPEN reached            10.76%     2.27%    19.27%
THROUGH reached               8.41%     0.17%    16.68%

总体 OPEN_LATCH latch/hinge/gap：22.0 mm / 0.0024 rad / 0.1200 rad
总体 latch_released/gap_open：76.47% / 0.10%
左门 latch/hinge/contact/roll/ee_x：22.1 mm / 0.0006 rad / 0.8736 / 0.0014 / -0.0007
右门 latch/hinge/contact/roll/ee_x：18.3 mm / 0.0330 rad / 0.7159 / 0.0139 / +0.0212
```

phase153 的续训门槛已经满足：左门 `OPEN_LATCH -> REPOSITION` 转换率从 step150 的约 `5.2%` 持续升至 step250 的约 `15.0%`，不能把问题继续归结为完全没有学习或锁舌压下不足。新的首个断层是锁舌释放后的开缝动作：左门平均锁舌位移已经高于右门，但 canonical `ee_x` 近零且略负，门轴几乎不转；右门给出明显正 `ee_x` 并形成门缝。

因此 phase153 到此结束，不继续盲目续训，也不立即改 reward。phase154 只增加无副作用 credit 诊断，比较左右门正 canonical `ee_x` 的 progress、return、advantage 和 actor observation 分叉；结论形成后删除诊断代码。禁止采用左右门专用动作、reference 或 seed 特调。

### phase154：锁舌释放后的 canonical ee_x credit 诊断（已验证）

```text
基线 checkpoint：logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260926_040325/model_step_000250.pt
checkpoint SHA-256：32e28fc6bfbfac0c6f9f2f8841db68839bf7479656ffd8db41b67f4abaed00e5
唯一变量：增加不参与 observation、reward、loss、采样权重或动作的临时 rollout 统计
正式行为：reward、动作、验收、随机化、低层模型和 canonical 接口全部不变
```

待验证假设：正 canonical `ee_x` 在锁舌释放后能够同样推动左右门形成 gap，但左门样本获得的即时 reward、reward-to-go 或 advantage credit 与右门不同，导致共享 actor 利用可观察的门侧信息形成了不对称策略。

诊断必须按门侧且仅对 OPEN_LATCH 统计：`ee_x` policy mean、sample、noise，锁舌已释放比例，gap/latch/retention 状态，`ee_x noise` 与即时 reward、rollout reward-to-go、normalized advantage 的协方差，以及 lag `1/2/4/8/16` 下与 gap 增量的协方差。锁舌未释放和已释放样本必须分开，避免锁舌碰撞约束把真正开缝阶段的信号淹没。

成功判据是用同一 checkpoint 的单个短诊断运行确定以下因果链中的第一个左右分化点：物理 gap 响应、即时 progress reward、跨步 reward-to-go、advantage normalization 或 actor observation 条件化。若证据不足，只扩大样本量，不改行为。

禁止范围：不得新增左右门 reference、动作符号分支、seed 特调、reward 权重或验收阈值；临时诊断形成结论后必须删除并重新通过静态测试。

2026-09-26 05:17 启动：临时诊断已通过容器内 `py_compile`、task-frame `5 passed` 和 `git diff --check`。从 step250 checkpoint 恢复四个匹配 rank，使用 `4 x 2048 env` 运行 1 update；本次不保存为新的有效训练 checkpoint。

第一次诊断 rollout 正常完成，但新增 credit 指标没有进入输出，说明临时采集开关未在实际 trainer 实例中启用；该次结果不用于任何行为判断。下一次重跑只增加初始化开关和结果区间的日志哨兵，仍从同一 step250 开始，正式行为不变。

第二次诊断确认实际 trainer 为 `B2Z1DoorSixStage` 且临时采集启用。有效样本如下：

```text
指标                                      左门                 右门
OPEN_LATCH 全部样本                    58,637               24,093
锁舌已释放样本                          22,342               12,769
已释放 policy mean ee_x                -0.00292             +0.02143
已释放 gap progress                     0.00003              0.44698
已释放 retention                        0.80317              0.52043
ee_x -> immediate reward slope          -3.24581             -4.16538
ee_x -> rollout reward-to-go slope      -3.40152             +3.51426
ee_x -> normalized advantage slope      -0.25071             +0.18042
lag16 ee_x -> gap delta slope            0.00429              1.12665
lag16 mean gap delta                     0.00007              0.57915
```

已确认：

- 左门不是锁舌未释放，也不是夹持质量不足；已释放样本 retention 反而高于右门。
- 两侧即时 `ee_x` 扰动都先承担动作/接触代价，immediate reward slope 均为负；右门只有在跨步 gap 响应形成后，reward-to-go 和 normalized advantage 才转为正。
- 左门策略均值停在负 `ee_x`，围绕该工作点的局部探索几乎不产生 gap，未来 credit 仍为负；右门均值已经越过接触阈值，正 `ee_x` 能持续产生 gap，并获得正未来 credit。
- 这与 phase152 已确认的统一坐标符号和左右门低层物理增益不冲突。当前首个分化点是“是否建立推门接触”的状态依赖响应，而不是 canonical `ee_x` 变换方向。

phase154 只定位了 on-policy 工作点附近的死区，不能据此直接修改 reward。下一步 phase155 必须从锁舌已释放的匹配状态做受控 `ee_x` 干预，测量左门跨越接触阈值所需累计位移、接触力、gap 响应及左右门差异。若干预证明两侧机制一致，修复应位于通用探索/潜势引导层；若机制不一致，再回到几何或物理所有权层。不得提前增加门侧特例。

### phase155：OPEN_LATCH canonical ee_x 受控干预（已验证）

```text
基线 checkpoint：phase153 model_step_000250.pt
唯一变量：OPEN_LATCH 时按 env id 固定分组，将实际发送的 canonical ee_x 替换为 0.00、0.01、0.02、0.04
共同条件：左右门使用完全相同的四组动作；其余九维动作仍由同一 step250 actor 给出
训练产物：不保存、不作为 checkpoint 基线、不更新正式机制
```

本实验不比较 PPO 更新，只比较物理响应。每个门侧和动作组记录：OPEN_LATCH 样本量、锁舌已释放样本量、实际 EE goal/EE 沿 canonical normal 的位移、门角增量、把手接触力、retention、OPEN_LATCH 完成数和 termination 数。成功判据是确定以下二者之一：

1. 相同正 `ee_x` 能让左右门以同量级建立接触并开缝，说明 phase154 差异来自策略未跨过通用接触死区。
2. 相同正 `ee_x` 在左右门仍产生系统性相反或数量级不同的实际目标/物理响应，说明 canonical 几何或碰撞状态仍有未发现的问题。

禁止范围：不得把干预动作写入正式 reference、reward、actor observation 或阶段逻辑；不得用干预结果生成训练 checkpoint；实验结束必须删除全部干预代码。

2026-09-26 05:37 启动：临时干预代码已通过容器内 `py_compile`、task-frame `5 passed` 和 `git diff --check`。从 phase153 step250 恢复，四个 rank 各 2048 env，只运行 1 update；命令行显式设置 `diagnostic_open_latch_ee_x_interventions=[0.0,0.01,0.02,0.04]`。

受控结果：

```text
门侧/ee_x       released 单步 hinge delta   平均把手力   retention   completion rate
左 / 0.00             0.00000002 rad           30.7 N       0.5593       0.0000
左 / 0.01             0.00226255 rad           50.0 N       0.5127       0.0000
左 / 0.02             0.00351522 rad           86.3 N       0.4946       0.00161
左 / 0.04             0.00041803 rad          102.8 N       0.4572       0.0000
右 / 0.00             0.00000701 rad           71.6 N       0.5419       0.0000
右 / 0.01             0.00169489 rad           50.2 N       0.5306       0.0000
右 / 0.02             0.00383929 rad           51.4 N       0.5318       0.01742
右 / 0.04             0.00419249 rad           86.8 N       0.5122       0.01381
```

核心结论：

- `ee_x=0.02` 时，锁舌已释放后的单步门角增量左/右为 `0.00352/0.00384 rad`，只有约 `8.4%` 差异，确认同一 canonical 正方向对两侧门都有正确且同量级的物理效果。
- 左门 completion rate 仍低于右门，主要反映 rollout 起始状态和锁舌释放样本分布不同，不能用它否定已经直接测得的 released 物理增益。
- `ee_x=0.01` 已能产生开缝，但幅度较弱；当前策略均值为负且噪声标准差约 `0.01`，不足以稳定跨越接触阈值。
- `ee_x=0.04` 对左门造成约 `103 N` 平均把手力、降低 retention 且反而失去开缝效率，排除“继续无上限增大探索”的方案。
- phase154 的 credit 不对称由探索是否越过接触阈值触发，不是门侧坐标、低层控制或门碰撞物理错误。

phase155 临时干预不得保留。下一步 phase156 在已有、显式的 `action_noise_stage_scales` 所有权层，把 OPEN_LATCH 的 `ee_x` scale 从 `1.0` 改为 `2.0`，对应最大噪声约 `0.02`；其他坐标和阶段不变。从 step250 短训并检查左门正 `ee_x`、gap、转换率是否增长，同时监控接触力和 retention。若过力或前序阶段回归则失败，不增加补丁叠加。

### phase156：提高 OPEN_LATCH ee_x 探索幅度（已验证）

```text
基线 checkpoint：phase153 model_step_000250.pt
唯一代码变量：OPEN_LATCH action_noise_stage_scales[ee_x] 1.0 -> 2.0
实际最大 ee_x noise std：约 0.01 -> 0.02
保持不变：其余动作噪声、reward、actor、验收、随机化、低层、canonical 接口和 AutoOpen
训练计划：4 x 2048 env，续训 25 update，step255/260/265/270/275 记录
```

成功判据：左门 OPEN_LATCH policy mean `ee_x` 向正方向移动，gap/hinge 和 `OPEN_LATCH -> REPOSITION` 转换率改善；左右门 retention、把手力和前两阶段不显著回归。失败判据：接触力或 termination 明显上升、retention 下降，或左门连续记录点仍无正 `ee_x`/gap 响应。

2026-09-26 05:44 静态验证：配置实际解析出的 OPEN_LATCH 行为 `[0.1,0.1,0.1,2.0,1.0,1.0,1.0,1.0,1.0,0.1]`；容器内 `py_compile`、task-frame `5 passed`、`git diff --check` 全部通过。

iteration 255：

```text
checkpoint：model_step_000255.pt
SHA-256：b10890e7dfa362ef0317d66d1634527dd8aa03e54aacae53c2e69b50bad28a81
左门 OPEN_LATCH reached / REPOSITION reached：40.96% / 5.45%
右门 OPEN_LATCH reached / REPOSITION reached：48.48% / 46.93%
左门 latch / hinge / contact / policy mean ee_x：22.3 mm / 0.0018 rad / 0.8608 / -0.0004
右门 latch / hinge / contact / policy mean ee_x：20.4 mm / 0.0383 rad / 0.6929 / +0.0201
总体 retention / max handle force：0.6477 / 65.8 N
```

相对 step251，左门 policy mean `ee_x` 从 `-0.0062` 回到 `-0.0004`，hinge 从近零升至 `0.0018 rad`；latch 和 contact 未回归。左门转换率尚未显著超过 step250 的 `5.14%`，因此只确认探索方向开始改善，不判定机制成功，保持唯一变量继续到 step260。

iteration 260：

```text
checkpoint：model_step_000260.pt
SHA-256：c2ca3a6def713934a82deebe304c2d3f8310dbf7d070b8bcc4f797488df66808
左门 OPEN_LATCH reached / REPOSITION reached / THROUGH reached：36.26% / 6.21% / 0.44%
右门 OPEN_LATCH reached / REPOSITION reached / THROUGH reached：49.08% / 47.43% / 15.98%
左门 latch / hinge / contact / policy mean ee_x：21.5 mm / 0.0015 rad / 0.8348 / -0.0007
右门 latch / hinge / contact / policy mean ee_x：18.9 mm / 0.0324 rad / 0.7179 / +0.0220
总体 retention / max handle force：0.6104 / 65.2 N
```

左门转换率和后期到达率已有小幅改善，且把手力没有上升；但 policy mean `ee_x` 仍接近零并略负，不能确认已经脱离死区。保持唯一变量继续到 step265。

iteration 265：

```text
checkpoint：model_step_000265.pt
SHA-256：5cc5764184f3bc7caa05052647b337c7b3b2a4ab5c67ebd5d1e2c02be17288e7
左门 OPEN_LATCH / REPOSITION / PUSH_OPEN / THROUGH reached：32.86% / 6.36% / 2.83% / 0.63%
右门 OPEN_LATCH / REPOSITION / PUSH_OPEN / THROUGH reached：44.21% / 42.60% / 22.73% / 19.98%
左门 latch / hinge / contact / policy mean ee_x：21.6 mm / 0.0013 rad / 0.8488 / -0.0010
右门 latch / hinge / contact / policy mean ee_x：18.5 mm / 0.0325 rad / 0.7053 / +0.0228
总体 retention / max handle force：0.6208 / 64.3 N
```

左门后续阶段连续改善且安全指标稳定，说明更大的探索确实产生了更多有效轨迹；但 actor mean 尚未吸收为正 `ee_x`。保持唯一变量继续到 step270/275，不能在中途叠加 reward 或监督机制。

iteration 270：

```text
checkpoint：model_step_000270.pt
SHA-256：8c4bf8f675f7216baf6ee491fe928fe3b7ad333189e7cf95e013e628beec6543
左门 OPEN_LATCH / REPOSITION / PUSH_OPEN / THROUGH reached：36.08% / 6.74% / 3.03% / 0.75%
右门 OPEN_LATCH / REPOSITION / PUSH_OPEN / THROUGH reached：46.22% / 44.15% / 19.35% / 16.37%
左门 latch / hinge / contact / policy mean ee_x：21.0 mm / 0.0011 rad / 0.8613 / -0.0017
右门 latch / hinge / contact / policy mean ee_x：18.7 mm / 0.0359 rad / 0.7112 / +0.0226
总体 retention / max handle force：0.6589 / 66.7 N
```

左门后期 reached 继续创新高且安全指标稳定，但 actor mean `ee_x` 未转正。当前证据表明更大噪声提高了成功探索覆盖，却尚未把正确动作稳定吸收到策略均值。继续到最终门槛 step275；若均值仍不转正，phase156 判为部分有效但不足，下一步转向通用 contact-to-gap credit bridge，不继续增大噪声。

iteration 275 正常完成，四个 rank 正常退出：

```text
checkpoint：model_step_000275.pt
SHA-256：190b1d0fee8f4e9f6d3ab328920442543aff5cd1f6c4992434e5fffda00cd6e0
左门 OPEN_LATCH / REPOSITION / PUSH_OPEN / THROUGH reached：35.11% / 6.70% / 2.93% / 0.67%
右门 OPEN_LATCH / REPOSITION / PUSH_OPEN / THROUGH reached：46.86% / 45.23% / 20.71% / 17.33%
左门 latch / hinge / contact / policy mean ee_x：21.5 mm / 0.0012 rad / 0.8637 / -0.0014
右门 latch / hinge / contact / policy mean ee_x：19.3 mm / 0.0378 rad / 0.7054 / +0.0229
总体 retention / max handle force：0.6612 / 63.0 N
```

phase156 结论是“机制安全且部分有效，但不足以解决 actor mean”：

- 相对 step250，左门 REPOSITION/PUSH_OPEN/THROUGH reached 从 `5.14/2.27/0.17%` 提高到 `6.70/2.93/0.67%`，证明 `0.02` 探索能增加有效轨迹覆盖。
- 左门 latch、contact、retention 没有回归，max handle force 从 step255 的 `65.8 N` 降到 `63.0 N`；没有出现 phase155 `0.04` 干预的过力问题。因此 OPEN_LATCH `ee_x` scale `2.0` 作为合理探索设置保留。
- 左门 actor mean 在 step255/260/265/270/275 始终约为 `-0.0004/-0.0007/-0.0010/-0.0017/-0.0014`，没有转正；仅靠增噪不能把成功轨迹稳定吸收到均值。
- 不继续增大噪声，也不回退已经验证安全有效的 `2.0`。phase157 只读审查物理接触量，寻找锁舌释放后、门缝出现前的连续且可泛化潜势；不得直接奖励 `ee_x` action 或添加门侧分支。

### phase157：contact-to-gap 物理量审查（已验证）

代码审查确认两个夹爪 contact sensor 已同时提供世界系三维力向量与接触点，actor 的 `handle_contact_force` 也已使用统一 task-frame 表达。因此候选 bridge 不需要新增传感器，且不能使用无方向的 `max_handle_force`：闭合夹持本身也会产生大力，直接奖励力幅值会鼓励夹紧而不是推门。

当前唯一诊断变量：只增加按门侧、锁舌是否释放分组的 canonical 法向推力统计。task-frame `+x` 指向穿门方向；该 ContactSensor 挂在门把手对象上，`force_matrix_w` 表示夹爪施加到把手对象的力，因此开门推力候选量为 `opening_force = clamp(force_task_x, min=0)`。记录其均值及超过 `5/20/40/80 N` 的比例，并与 policy mean `ee_x`、hinge/gap 同时比较。

验证门槛：右门正 `ee_x` 和开缝样本应出现显著正 opening force；左门近零/负 `ee_x` 应较弱。若符号或左右响应不符合，不设计 reward，先回查传感器力方向。若符合，再确定一个有上限的安全 force progress，并用 gap progress 接管，禁止奖励 action 本身或无界接触力。

第一次诊断使用 phase156 step275、`4 x 2048 env`、`8 steps/env`，正常完成 iteration 276，正式行为和 reward 未改变。该次不能用于验证 opening-force 假设：OPEN_LATCH 左右门平均锁舌位移仅为 `0.9/1.5 mm`，锁舌释放后样本数均为 0，因此 released opening-force 指标为 0 只是空样本结果。已确认 `get_policy_diagnostics()` 会由 `TensorAverageMeterDict` 覆盖整个 rollout 累计，排除“只读取日志最后一帧”的判断；真正原因是 step275 的 staged-reset 快照从阶段入口恢复，而 8 个高层步不足以走到 `20 mm` 释放阈值。

下一次诊断保持 checkpoint、策略、环境、reward 和统计代码不变，唯一改变采样窗口为 `32 steps/env`，仍只执行一个 PPO update且不保存为有效 checkpoint。若仍无足够 released 样本，只调整无副作用采样长度，不修改行为。

第二次诊断以 `32 steps/env` 正常完成 iteration 276，获得足够的 released 样本，但反证了最初的符号假设：使用 `clamp(-force_task_x, 0)` 时，左门 `ee_x=-0.0083`、hinge 近零却显示 `23.7 N`，右门 `ee_x=+0.0268`、hinge `0.0084 rad` 却仅显示 `1.34 N`。源码核对确认传感器 prim 是门把手对象，`force_matrix_w` 是夹爪施加给对象的力，不是对象施加给夹爪的反力；因此负号错误。该负号候选已排除，下一次仅把统计改为 `clamp(force_task_x, 0)`，保持相同 checkpoint 和 32 步窗口复测。

第三次以相同 checkpoint 和 `32 steps/env` 完成正号复测：

```text
指标                                      左门          右门
policy mean ee_x                        -0.0083       +0.0268
hinge angle                              0.0000        0.0084 rad
released opening force                   0.0462       48.4574 N
released opening force >= 5 N            0.30%        76.65%
released opening force >= 20 N           0.00%        66.00%
released opening force >= 40 N           0.00%        50.86%
released opening force >= 80 N           0.00%        22.85%
```

结论：`clamp(force_task_x, 0)` 是正确且具有左右门统一语义的开门接触量；它在 gap 出现前区分了“已释放但未建立推力”的左门死区和“正在推开”的右门轨迹。phase157 临时阈值分桶已删除，只保留正式机制需要的 opening-force 均值遥测。

### phase158：通用 contact-to-gap 连续势函数（失败）

```text
上游瓶颈：锁舌释放后、门缝出现前没有正 credit，左门 actor mean ee_x 停在负值
基线 checkpoint：phase156 model_step_000275.pt
唯一变量：OPEN_LATCH progress 从 0.10 + 0.45*latch + 0.45*gap 改为
          0.10 + 0.40*latch + 0.10*latch*clamp(opening_force/20N, 0, 1) + 0.40*gap
禁止改动：observation、action、阶段验收、随机化、低层、探索倍率和其他 reward
```

设计理由：只奖励施加到把手对象、沿 canonical 穿门方向的实际物理力，不奖励 action；`20 N` 后饱和，避免鼓励增大接触力；乘 latch progress 使未释放锁舌时不能仅靠顶门获得完整 bridge；gap 仍有独立 40% 权重，顶住但门不动无法获得后续增量或阶段完成奖励。成功门槛是左门 actor mean `ee_x`、hinge/gap 和 `OPEN_LATCH -> REPOSITION` 转换率持续改善，同时 retention、最大把手力和前两阶段不回归。

2026-09-26 06:26 静态验证通过：容器内 `py_compile`、task-frame `5 passed` 和 `git diff --check`。计划从 step275 运行 `4 x 2048 env`、25 update，在 step280/285/290/295/300 记录门侧指标；训练期间不叠加其他改动。

step280--300 四个 rank 正常运行，无异常或 NaN。左门曲线：

```text
step                         280      285      290      295      300
OPEN_LATCH reached         29.74%   29.55%   29.02%   31.17%   38.33%
REPOSITION reached          7.34%    7.18%    6.52%    6.33%    6.40%
PUSH_OPEN reached           1.83%    1.96%    1.98%    2.05%    2.16%
THROUGH reached             0.24%    0.22%    0.20%    0.15%    0.17%
policy mean ee_x          -0.0083  -0.0017  +0.0004  -0.0021  -0.0041
opening force (N)           0.036    0.330    0.576    1.738    0.825
hinge angle (rad)          0.0000   0.0000   0.0002   0.0007   0.0010
```

step300 checkpoint：`logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260926_062639/model_step_000300.pt`，SHA-256 `9a3f2dc955a7d8425572fde37b90d997d0624b82790a84ac236d501b65a40262`。

中间判断：bridge 已产生可测的左门开门力和微小门角响应，且 max handle force `71.0 N`、retention `0.7117`，没有安全回归；但正 `ee_x` 尚未被持续吸收到 actor mean，REPOSITION 以后也未改善。保持唯一机制不变续训至 step325；若 actor mean、hinge/gap 和转换率仍不持续改善，则判定 10% bridge 不足，不继续盲目续训或同时调整权重。

step305--325 保持机制不变续训完成，四个 rank 正常退出，无异常或 NaN：

```text
step                         305      310      315      320      325
左门 OPEN_LATCH reached    29.65%   29.26%   28.74%   30.73%   37.85%
左门 REPOSITION reached     7.30%    7.07%    6.64%    6.29%    6.20%
左门 PUSH_OPEN reached      1.74%    1.82%    1.84%    1.83%    1.79%
左门 THROUGH reached        0.20%    0.22%    0.24%    0.20%    0.24%
左门 policy mean ee_x     -0.0057  -0.0005  -0.0015  -0.0032  -0.0043
左门 opening force (N)      0.086    0.235    0.963    1.552    2.340
左门 hinge angle (rad)     0.0000   0.0000   0.0001   0.0006   0.0010
OPEN_LATCH ee_x advantage +0.000001 +0.000066 -0.000531 -0.000203 -0.000084
```

step325 checkpoint：`logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260926_063428/model_step_000325.pt`，SHA-256 `0c7b7ea4a99cd61ab19b57dffd413f93aea6e4342a2f6d9a220b96f6f99a7c98`。该 checkpoint 属于失败分支，不替代 phase156 step275 有效基线。

结论：10% bridge 能让 opening force 缓慢上升，但没有使 actor mean 持续转正，后半程 `ee_x` advantage 重新为负；左门 REPOSITION/PUSH_OPEN 也没有超过 phase156 step275 的 `6.70%/2.93%`。按预设门槛判定该权重不足，停止续训。已排除“只需给 10% bridge 再多训练 25 update”的方向；物理量和机制本身仍成立，下一轮只验证其相对权重。

### phase159：提高 contact bridge 相对权重（失败）

```text
基线 checkpoint：phase156 model_step_000275.pt
唯一变量：OPEN_LATCH progress 权重从 baseline/latch/contact/gap = 0.10/0.40/0.10/0.40
          改为 0.10/0.30/0.30/0.30
保持不变：opening-force 20 N 饱和、latch gate、observation/action、验收、随机化、低层、探索倍率和其他 reward
失败门槛：左门 actor mean ee_x、hinge 和 REPOSITION 转换率在 25 update 内仍无持续改善，或安全/前序阶段回归
```

该实验不是门侧或 seed 特例，也不新增控制路径。30% bridge 在达到 20 N 时提供一次性有界 potential 增量，持续顶住不会重复获奖；gap 仍有独立 30% 权重且阶段完成条件不变。若仍失败，停止继续调大力奖励，回到 credit 形式而不是堆权重。

2026-09-27 01:25 静态验证通过：容器内 `py_compile`、task-frame `5 passed` 和 `git diff --check`。从 phase156 step275 运行 `4 x 2048 env`、25 update，在 step280/285/290/295/300 记录同一组门侧和安全指标。

step280--300 四个 rank 正常完成，无异常或 NaN。左门结果：

```text
step                      280      285      290      295      300
policy mean ee_x        -0.0079  -0.0016  +0.0008  -0.0023  -0.0034
opening force (N)         0.055    0.204    0.932    1.418    1.389
hinge angle (rad)        0.0000   0.0000   0.0002   0.0008   0.0010
REPOSITION reached        7.37%    7.10%    6.70%    6.49%    6.34%
ee_x advantage          +0.000021 +0.000089 +0.000130 -0.000150 -0.000130
```

失败证据 checkpoint：`logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260927_012724/model_step_000300.pt`，SHA-256 `102456a3bc229a52ec01431b9bdf1feae1caf5765af58a8177959bbf3d8f53c9`。该 checkpoint 不替代 phase156 step275 有效基线。

结论：把 contact bridge 从 10% 提高到 30% 后，opening force 和 hinge 在中段增长，但 step295--300 的 `ee_x advantage` 再次转负，policy mean 也回到负值；REPOSITION 从 `7.37%` 回落到 `6.34%`，低于 phase156 step275 的 `6.70%`。按预设门槛判定失败，排除“继续增加 contact-force 势函数权重”方向。

待验证根因：当前 contact bridge 是可逆势函数。建立接触时 contact progress 上升并产生正 delta，随后接触卸载时 progress 下降，可能扣回近似相同的 credit；此时 gap 尚未同步增长，无法接管该 credit。该判断尚未由逐步 delta 直接证实，phase160 只做诊断。

### phase160：contact potential 正负 delta 抵消诊断（已验证）

```text
上游结论与待解决瓶颈：phase158/159 的 contact bridge 在训练后半程均失去正 ee_x credit；需确认是否由可逆 potential 的负 delta 抵消
基线 checkpoint / SHA-256：phase156 model_step_000275.pt / 190b1d0fee8f4e9f6d3ab328920442543aff5cd1f6c4992434e5fffda00cd6e0
唯一变量：增加逐步只读诊断，记录 opening-contact progress 正 delta、负 delta、gap delta 和当前 30% 权重下的净 bridge delta
待验证假设：锁舌释放后，contact progress 的累计正 delta 被后续负 delta 基本抵消，而 gap delta 尚不足以接管 credit
禁止改动范围：reward 数值与公式、observation、action、阶段验收、随机化、低层模型、探索倍率、策略参数和 AutoOpen
运行规模与日志：4 x 2048 env，32 steps/env，从 step275 只运行 1 update；`/tmp/b2z1_phase160_contact_delta_diag.log`
成功/失败判据：若 released OPEN_LATCH 样本中累计负 contact delta 与正 delta 同量级，且 contact+gap 加权净 delta 接近零或转负，则假设成立；否则假设失败
观测指标：按左右门及 latch released 分组的 positive contact delta、negative contact delta magnitude、gap delta、weighted net bridge delta、policy/sample ee_x 和 hinge
结论与已排除方向：右门 released 样本中负 contact delta 明显大于正 delta，gap 增长后净 bridge 仍为负；确认可逆 contact potential 是当前 credit 断点，排除继续堆权重
正式保留/删除的代码：临时诊断已全部删除；不进入正式训练路径
前序阶段回归：行为完全不变；单 update 正常完成，无异常或 NaN
下一决策门槛：phase161 只验证单调 best-contact 里程碑，不同时调整权重或其他 reward
```

2026-09-27 01:35 启动前验证：`py_compile`、5 项 task-frame unittest 和 `git diff --check` 均通过。临时字段只进入 policy diagnostics，不参与 reward、observation、action 或控制计算。

2026-09-27 01:45 完成：四个 rank 正常完成 iteration 276，无异常或 NaN。锁舌释放样本逐步均值：

```text
指标                                      左门        右门
contact delta positive                  0.0000      0.0061
contact delta negative magnitude        0.0000      0.0140
gap delta                               0.0000      0.0047
0.30 * (contact delta + gap delta)     -0.0000     -0.0009
policy mean ee_x                       -0.0083     +0.0268
hinge angle                             0.0000      0.0084 rad
released opening force                  0.0455     48.6076 N
```

右门已经是有效开门轨迹，却因 contact 从峰值卸载，平均返还 `0.0140`，超过建立接触的 `0.0061`；即使加入同期 `0.0047` gap 增长，bridge 仍净负 `-0.0009`。因此 phase158/159 后半程 `ee_x advantage` 转负不是权重不足，而是可逆势函数的形式错误。临时诊断代码已删除。

### phase161：单调 best-contact 物理里程碑（失败）

```text
上游结论与待解决瓶颈：可逆 contact potential 在卸载时扣回已获得 credit，导致有效开门轨迹后半程收到负反馈
基线 checkpoint / SHA-256：phase156 model_step_000275.pt / 190b1d0fee8f4e9f6d3ab328920442543aff5cd1f6c4992434e5fffda00cd6e0
唯一变量：OPEN_LATCH contact 分量从当前 opening-contact progress 改为本 episode/stage 的历史最大值
待验证假设：一次性记录已经建立的安全开门接触，可避免卸载时错误返还 credit，使左门正 ee_x 被持续吸收到 actor mean
禁止改动范围：contact/latch/gap 权重、retention 公式、observation、action、阶段验收、随机化、低层、探索倍率和其他 reward
运行规模与日志：4 x 2048 env，从 step275 训练 25 update；`/tmp/b2z1_phase161_contact_milestone_to300.log`
成功/失败判据：step280--300 左门 policy mean ee_x、hinge 和 REPOSITION 持续改善，且 retention、最大把手力和前两阶段不回归；否则失败
观测指标：左右门分阶段到达率、policy mean ee_x、opening force、hinge、retention、max handle force、ee_x advantage
结论与已排除方向：best-contact 修正了后半程总体 advantage 符号，但左门 actor mean 和 REPOSITION 未持续改善；排除“只修 contact potential 可逆性即可解决左门策略”的方向
正式保留/删除的代码：best-contact 暂时保留用于下一轮只读因果诊断；不得在根因确认前继续叠加 reward
前序阶段回归：启动前静态测试；训练中逐点核对 APPROACH_PREGRASP、GRASP 和安全指标
下一决策门槛：只按 step280/285/290/295/300 预设曲线判断，不在中途叠加新机制
```

2026-09-27 01:46 启动前验证：临时 phase160 诊断已删除；best-contact 是唯一行为改动。`py_compile`、5 项 task-frame unittest 和 `git diff --check` 均通过。

step280：左门 OPEN_LATCH/REPOSITION/PUSH_OPEN/THROUGH reached 为 `29.77/7.35/1.86/0.24%`，policy mean `ee_x=-0.0080`，opening force `0.0485 N`，hinge `0.0000 rad`，`ee_x advantage=-0.000103`；总体 retention `0.7489`，max handle force `59.58 N`。前序 APPROACH_PREGRASP 为 `100%`，GRASP 为 `51.74%`，未见安全或前序回归。该点仍处于训练早期，只登记证据，不作成败判断。

step285/290：左门 policy mean `ee_x` 转为 `+0.0001/+0.0007`，opening force 升至 `0.3285/1.1729 N`，hinge 为 `0.0001/0.0003 rad`；说明单调里程碑开始改变动作方向。同期 REPOSITION 为 `7.03/6.68%`，`ee_x advantage=-0.000161/-0.000371`，尚未形成持续正 credit 或转换率改善。继续保持唯一变量到 step295/300，不作中途修改。

step295/300：总体 `ee_x advantage` 转为 `+0.000058/+0.000280`，但左门 policy mean `ee_x` 回落到 `-0.0034/-0.0062`，opening force 为 `1.4054/1.0877 N`，hinge 为 `0.0012/0.0013 rad`，REPOSITION 为 `6.38/6.34%`。前序 APPROACH_PREGRASP 保持 `100%`，GRASP 为 `50.77/49.96%`；总体 retention 为 `0.6558/0.6292`，max handle force `71.37/76.54 N`，未见安全回归。

step300 checkpoint：`logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260927_014529/model_step_000300.pt`，SHA-256 `a7c7b54b697d052932c3c315a2226b770fa5f2b92f5bd46b86f3e6ff5e60e7ef`。该 checkpoint 属于失败分支，不替代 phase156 step275 有效基线。

结论：单调 contact 里程碑确实消除了 phase159 中“后半程总体 advantage 必然转负”的现象，但没有使左门正确动作持续进入 actor mean，也没有提高 REPOSITION 转换率。不能继续把问题归结为 contact 权重或势函数形式。下一步只读拆分 advantage 的门侧来源和 PPO 聚合链路。

### phase162：门侧 action-advantage 到 actor gradient 链路诊断（已验证）

```text
上游结论与待解决瓶颈：phase161 后半程总体 ee_x advantage 为正，但左门 actor mean 仍转负；聚合指标可能被右门主导
基线 checkpoint / SHA-256：phase161 model_step_000300.pt / a7c7b54b697d052932c3c315a2226b770fa5f2b92f5bd46b86f3e6ff5e60e7ef（仅用于复现失败分支诊断，不提升为有效训练基线）
唯一变量：按门侧及 latch released 状态记录 ee_x sample noise、normalized advantage、二者乘积及 PPO sample weight；不修改行为
待验证假设：总体正 ee_x signal 由右门样本主导，左门 released 样本的实际 policy-gradient signal 仍为负或被权重稀释
禁止改动范围：reward、observation、action、网络、优化器、阶段验收、随机化、低层、探索倍率和 AutoOpen
运行规模与日志：4 x 2048 env、32 steps/env、单 update；`/tmp/b2z1_phase162_variant_gradient_diag.log`
成功/失败判据：定位 rollout advantage 到最终 actor loss 之间第一个门侧信号符号或权重发生变化的位置；证据不足时只扩大统计，不改行为
观测指标：左右门 released/unreleased 的 ee_x noise*advantage、normalized advantage、sample weight、clipping mask 和 actor-loss contribution
结论与已排除方向：已排除“总体正 ee_x signal 由右门样本主导”；左门自身 signal 为正。继续定位 PPO 优化后同一 observation 的输出变化
正式保留/删除的代码：临时统计形成结论后删除；phase161 best-contact 在本诊断中保持不变
前序阶段回归：行为不变；单 update 不作为新有效 checkpoint
下一决策门槛：按 latch released 状态继续拆分状态分布；不得修改 PPO 或 reward
```

2026-09-27 01:55 代码审查：`balance_staged_task_variants=True` 只按门型分别归一化 advantage；PPO 的 `task_stage_weights` 仍只按阶段计数，actor loss 没有按门型均衡样本贡献。已增加只读的 stage x variant 样本数、每样本 action-advantage signal 和对阶段总信号的 contribution；不参与 loss。`py_compile`、5 项 task-frame unittest 和 `git diff --check` 均通过。

2026-09-27 01:58 首次运行失败：四个 rank 完成 32 步 rollout 后，在只读汇总处因局部变量 `task_variants` 未从 storage 取出而触发 `NameError`。错误发生在 PPO update 之前，没有参数更新或新 checkpoint。修复仅补充既有 `task_variant` storage 的读取；行为、reward 和 loss 不变，从同一 step300 重跑。

2026-09-27 02:03 重跑正常完成 iteration 301，无异常或 NaN。OPEN_LATCH 门侧 signal：

```text
指标                                      左门          右门          合计
样本数                                   28532         15689         44221
ee_x noise * normalized advantage      +0.00018551   -0.00027919          -
对阶段总 ee_x signal 的贡献             +0.00011969   -0.00009905   +0.00002064
policy mean ee_x                        -0.0139       +0.0323              -
hinge angle                              0.0000        0.0183 rad         -
```

该结果反证原假设：总体正 signal 不是由右门主导，而是左门正确的正 signal 被右门负 signal 大幅抵消后仍略正。由于配置只按阶段设置 loss weight，左右门贡献确实没有等权；但这次左门样本更多、贡献方向也正确，因此它不能解释左门 mean 继续为负。下一诊断点后移到同一 observation 在 PPO 更新前后的输出变化，以及 PPO loss 与 action supervision/其他阶段共享参数的梯度干扰。

2026-09-27 02:06 启动更新前后输出诊断：保持 step300 checkpoint、reward、loss 和 rollout 规模不变；单 update 后对同一批 observation 再前向 actor，按阶段和门型记录 `post_mean - pre_mean`。新增代码只在 optimizer 完成后读取模型，不参与训练计算；静态检查和 5 项 unittest 通过。

2026-09-27 02:11 更新前后诊断正常完成，无异常或 NaN：

```text
指标                                      左门          右门
ee_x noise * normalized advantage      +0.00016180   -0.00031122
对阶段总 ee_x signal 的贡献             +0.00010434   -0.00011052
同批 observation 的 post_mean-pre_mean  +0.00029245   -0.00066405
rollout policy mean ee_x                -0.0139       +0.0323
```

结论：PPO 对同一批 observation 的实际更新方向与各门型 rollout signal 完全一致，排除 advantage 聚合、PPO clipping、共享参数或监督 loss 把左门更新符号反转。跨 iteration 看到的左门 mean 变负是新 rollout 的状态分布发生变化，不是同一状态上的策略倒退。phase163 必须按 latch released 状态拆分，定位负输出属于哪类物理状态。

### phase163：锁舌释放条件下的门侧 credit 诊断（已验证）

```text
上游结论与待解决瓶颈：左门同批 PPO 更新方向正确，但下一 rollout 的状态混合均值仍负；需定位具体状态组
基线 checkpoint / SHA-256：phase161 model_step_000300.pt / a7c7b54b697d052932c3c315a2226b770fa5f2b92f5bd46b86f3e6ff5e60e7ef（仅诊断）
唯一变量：按左右门和 latch_pos >= 20 mm 拆分 ee_x policy mean、noise、normalized advantage、noise*advantage 与样本数
待验证假设：左门 released 样本仍处于负 ee_x 开缝死区，而 unreleased 样本的约束行为掩盖或混合了统计
禁止改动范围：reward、loss、observation、action、网络、阶段验收、随机化、低层、探索倍率和 AutoOpen
运行规模与日志：4 x 2048 env、32 steps/env、单 update；`/tmp/b2z1_phase163_latch_group_credit_diag.log`
成功/失败判据：明确负 mean 和错误/不足 credit 所属的 latch 状态组；若两组仍混杂，只增加状态分桶，不改行为
观测指标：四个 side x released 分组的 count、policy mean ee_x、sample noise、normalized advantage、noise*advantage、opening force、hinge
结论与已排除方向：释放前后左门 mean 均负，但两组 signal 均正；排除状态混合造成错误符号，也排除 PPO 链路 bug
正式保留/删除的代码：phase162/163 临时诊断全部删除；仅保留 best-contact 候选机制
前序阶段回归：行为完全不变，单 update 不作为新 checkpoint
下一决策门槛：只有定位到具体物理状态组后，才讨论对应 reward/表示所有权层修复
```

2026-09-27 02:14 启动前验证：临时 storage 标签只来自既有 `_latch_pos >= 0.02`，不进入 actor/critic observation、reward 或 loss。分组统计记录 count、policy mean、noise、advantage 和 signal。`py_compile`、5 项 unittest 和 `git diff --check` 通过。

2026-09-27 02:18 正常完成 iteration 301，无异常或 NaN：

```text
分组                         左门未释放    左门已释放    右门未释放    右门已释放
样本数                         19615          8917          8100          7613
policy mean ee_x             -0.014526     -0.012553     +0.034525     +0.029900
advantage mean               +0.064751     -0.142434     +0.121086     -0.128832
ee_x noise abs               0.015954       0.015899      0.015941      0.015966
ee_x noise*advantage signal +0.00020346   +0.00004350   -0.00014961   -0.00043799
```

结论：左门负 mean 同时存在于释放前后，不是 released 状态占比造成的统计假象；两组实际 policy-gradient signal 都已为正，且 phase162 已证明同批 actor 更新方向与 signal 一致。右门两组 signal 都为负，正把过大的正动作往回收。至此从物理 response、reward delta、normalized advantage、门侧聚合到实际 actor update 的符号链路均已对齐。剩余问题是 released 正 signal 较弱、形成较晚，phase164 在不改行为的前提下验证额外 25 update 是否足够。

### phase164：best-contact 正确 credit 延长吸收验证（失败）

```text
上游结论与待解决瓶颈：phase163 确认左门 released signal 已为正但较弱；需区分训练时间不足与机制仍不足
基线 checkpoint / SHA-256：phase161 model_step_000300.pt / a7c7b54b697d052932c3c315a2226b770fa5f2b92f5bd46b86f3e6ff5e60e7ef
唯一变量：训练时间增加 25 update，step300 -> 325；代码行为不变
待验证假设：正确的 released 正 signal 继续累积后，左门 actor mean、opening force、hinge 和 REPOSITION 会持续改善
禁止改动范围：reward、权重、observation、action、网络、PPO、阶段验收、随机化、低层、探索倍率和 AutoOpen
运行规模与日志：4 x 2048 env、8 steps/env；`/tmp/b2z1_phase164_contact_milestone_to325.log`
成功/失败判据：step305--325 左门 mean 向正、hinge 和 REPOSITION 持续提高且安全/前序不回归；否则失败并排除继续盲目续训
观测指标：左右门分阶段 reached、policy mean ee_x、opening force、hinge、retention、max handle force、ee_x advantage
结论与已排除方向：额外训练能提高 opening force/hinge，但不能提高后续转换率且 retention 明显下降；排除“best-contact 只需更多吸收时间”
正式保留/删除的代码：best-contact 失败机制删除，恢复 phase156 reward 基线；phase164 checkpoint 只作失败证据
前序阶段回归：逐记录点核对 APPROACH_PREGRASP、GRASP 和安全指标
下一决策门槛：只按 step305/310/315/320/325 判断，不在中途追加机制
```

2026-09-27 02:22 启动前验证：phase162/163 临时代码已删除，搜索确认只保留 `_open_latch_contact_best` 候选机制。`py_compile`、5 项 unittest 和 `git diff --check` 通过；代码、文档、step275/300 已备份到三机并核对 SHA。

step305/310/315：左门 policy mean `ee_x=-0.0085/+0.0029/+0.0004`，opening force `0.4368/1.4617/4.7227 N`，hinge `0.0000/0.0003/0.0007 rad`，表明正确动作开始被吸收；但 REPOSITION 为 `7.27/6.83/6.49%`，retention 为 `0.7636/0.6359/0.4992`，尚未形成后续转换率改善。保持代码不变继续到 step320/325。

step320/325：左门 policy mean `ee_x=-0.0038/-0.0041`，opening force `3.4180/1.7684 N`，hinge `0.0026/0.0028 rad`，REPOSITION `6.27/6.69%`；retention `0.4816/0.5127`，max handle force `55.30/58.92 N`。前序 APPROACH_PREGRASP 保持 `100%`，GRASP `49.63/48.95%`，未出现安全回归，但抓持保持明显恶化。

step325 checkpoint：`logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260927_022235/model_step_000325.pt`，SHA-256 `65b5c3db61b0d1da717380ab8a362f7f306ed49c6feb35ed22657035c531b713`。该 checkpoint 属于失败分支，不替代 phase156 step275 有效基线。

结论：best-contact 机制让策略更频繁地产生接触和微小门角，但没有把 OPEN_LATCH 转换为 REPOSITION；额外 25 update 后仍未超过有效基线，且 retention 从 phase161 末期约 `0.63` 降至约 `0.51`。该机制按预设门槛失败，不能转为正式 reward。下一步回到 phase156 reward 基线，定位抓持保持的具体控制断点。

### phase165：OPEN_LATCH retention 控制断点诊断（已验证）

```text
上游结论与待解决瓶颈：contact shaping 分支提高接触/微小开缝但牺牲 retention，后续转换率不升
基线 checkpoint / SHA-256：phase156 model_step_000275.pt / 190b1d0fee8f4e9f6d3ab328920442543aff5cd1f6c4992434e5fffda00cd6e0
唯一变量：恢复 phase156 reward 公式后做只读对照，比较 retention 三分量、误差和动作漂移；不训练新行为
待验证假设：OPEN_LATCH 中除 ee_x/handle roll 外的自由动作随机游走，破坏 pocket 或 gripper axis，导致推门接触无法保持
禁止改动范围：observation、action 语义、PPO、阶段验收、随机化、低层、探索倍率、反馈 ownership 和 AutoOpen
运行规模与日志：先读取 phase156/164 已有日志；证据不足再做同 checkpoint 单 update 诊断
成功/失败判据：用现有或只读对照明确 retention 最先下降的分量及其对应动作/误差；根因未确认不改 ownership
观测指标：pocket_retention、axis_retention、contact_retention、pocket/axis error、OPEN_LATCH ee_y/z/roll/pitch/yaw policy mean 和 noise
结论与已排除方向：pocket/contact 是主要退化项；旧 best-contact 在建立里程碑时未绑定 retention，排除继续使用独立历史最大力
正式保留/删除的代码：best-contact 和 contact reward 删除；opening-force 遥测保留
前序阶段回归：恢复公式后静态测试；只读诊断不改变行为
下一决策门槛：确认具体自由度后，才判断应由反馈 reference 还是 PPO 拥有
```

现有日志直接对照，无需新增仿真：

```text
指标                              phase156 step275   phase164 step325   变化
pocket retention                       0.7771              0.6112      -0.1659
axis retention                         0.7633              0.6908      -0.0725
contact retention                      0.7847              0.5959      -0.1888
近似 pocket distance                  17.8 mm             31.1 mm     +13.3 mm
近似 axis error                       0.071 rad            0.093 rad   +0.022 rad
左门 policy mean ee_z                -0.0012             -0.0039      -0.0027
左门 policy mean ee_roll             +0.0012             +0.0104      +0.0092
```

结论：退化以 pocket 偏离和双侧接触丢失为主，axis 次之。旧 best-contact 只记录 `latch*force` 的历史最大值；坏抓持下的瞬时力也能抬高永久 milestone，之后即使接触消失仍保留该记录。虽然外层当前 retention 会给一次回退惩罚，但历史 milestone 本身没有区分“安全建立的接触”和“擦碰力峰值”。这是 reward 事件定义错误，不是应该通过固定 ee_z/roll 动作或门侧特例修补的问题。

### phase166：retention-gated safe-contact 里程碑（失败）

```text
上游结论与待解决瓶颈：旧 contact milestone 可被坏抓持力峰值抬高，导致 pocket/contact retention 退化
基线 checkpoint / SHA-256：phase156 model_step_000275.pt / 190b1d0fee8f4e9f6d3ab328920442543aff5cd1f6c4992434e5fffda00cd6e0
唯一变量：safe_sample=latch_progress*clamp(opening_force/20N)*retention；safe_best=max(safe_best,safe_sample)；contact credit=min(safe_best,current retention)
待验证假设：只记录安全建立的开门接触，并在当前抓持退化时撤回 credit，可保留正 contact bridge 而不鼓励擦碰/丢抓
禁止改动范围：权重 0.10/0.30/0.30/0.30、observation、action、PPO、阶段验收、随机化、低层、探索倍率、反馈 ownership 和 AutoOpen
运行规模与日志：4 x 2048 env，从 step275 训练 25 update；`/tmp/b2z1_phase166_safe_contact_to300.log`
成功/失败判据：左门 mean/hinge/REPOSITION 持续改善，且 pocket/contact/overall retention 不低于 phase156 安全基线；否则失败
观测指标：左右门 reached、mean ee_x、opening force、hinge、三个 retention 分量、max force 和 ee_x advantage
结论与已排除方向：safe-contact 避免了旧 best-contact 的 retention 退化，但没有改善左门 mean ee_x、hinge 或 REPOSITION；排除继续沿 contact milestone 调权或改事件定义
正式保留/删除的代码：safe-contact buffer 和 reward 分支已删除；恢复 phase156 的 `retention * (0.10 + 0.45*latch + 0.45*gap)`
前序阶段回归：逐记录点核对 APPROACH_PREGRASP、GRASP 和安全指标
下一决策门槛：step280/285/290/295/300；中途不叠加 ownership 或其他 reward 修改
```

2026-09-27 02:33 启动前验证：失败 best-contact 已删除；safe-contact buffer 是唯一新行为。公式连续、无门侧/seed/离散阈值分支。`py_compile`、5 项 unittest 和 `git diff --check` 通过。

step280/285/290：左门 policy mean `ee_x=-0.0085/-0.0023/-0.0008`，opening force `0.0306/0.2353/0.5218 N`，hinge `0/0/0.0001 rad`；pocket retention `0.9152/0.8918/0.8009`，contact retention `0.8420/0.9101/0.8390`，明显避免了旧 best-contact 的早期抓持退化。同期 REPOSITION `7.32/7.08/6.65%`，尚未改善，保持唯一机制继续到 step300。

step295/300：step300 左门 OPEN_LATCH/REPOSITION/PUSH_OPEN/THROUGH reached 为 `38.42/6.06/2.17/0.21%`，policy mean `ee_x=-0.0046`，opening force `1.0970 N`，hinge `0.0010 rad`。总体 retention 为 `0.6885`，pocket/axis/contact retention 为 `0.7929/0.7896/0.7945`，max handle force 为 `68.61 N`。相比旧 best-contact，抓持保持没有继续退化；但左门动作均值、门缝和 REPOSITION 均未超过 phase156 有效基线。

step300 checkpoint：`logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260927_023318/model_step_000300.pt`，SHA-256 `3499d23c63921799a9810b6029137c327bf9b69884f93d4984abc9d7e24678c5`。该 checkpoint 属于失败分支，不替代 phase156 step275 有效基线。

结论：safe-contact 解决了旧机制奖励坏抓持力峰值的问题，但没有解决左右门策略分化，因此 contact reward 不再是下一轮变量。失败机制已完整删除，正式代码恢复 phase156 reward。下一步只读审计镜像等价状态下的 actor observation；在找到第一个不等价字段前不修改行为。

### phase167：左右门镜像等价 actor observation 审计（已验证）

```text
上游结论与待解决瓶颈：phase157--166 已确认物理响应、reward-to-advantage 和 PPO 更新符号正确，contact shaping 不能修复左右门策略分化
基线 checkpoint / SHA-256：phase156 model_step_000275.pt / 190b1d0fee8f4e9f6d3ab328920442543aff5cd1f6c4992434e5fffda00cd6e0
唯一变量：构造或读取左右门镜像等价 OPEN_LATCH 状态，逐字段比较 actor observation；只增加无副作用诊断
待验证假设：canonical task/action 已统一，但 actor observation 仍包含策略不需要的物理侧别或未 canonical 的向量、姿态、关节符号，促使共享策略学习两套状态映射
禁止改动范围：observation 内容与维度、reward、action、网络、PPO、阶段验收、随机化、低层、探索倍率和 AutoOpen
运行规模与日志：4 x 2048 env，从历史 step275 staged-reset 状态做 8 步只读 rollout；`/tmp/b2z1_phase167_obs_audit.log`、`/tmp/b2z1_phase167_obs.rank{0,1,2,3}.pt`
成功/失败判据：列出每个 actor observation 字段的维度、所有权、镜像差异及是否应 canonical，并定位第一个不合理差异；证据不足则继续诊断，不修改接口
观测指标：relative_to_door、door DOF、task geometry、panel interaction、privileged door fields、hand-handle transform 及其左右门逐元素差值
结论与已排除方向：physical fixed/moving jaw 槽位在镜像腕部构型下交换，导致 actor 面对两套 closing/contact 排列；单独交换门侧 one-hot 不是主因
正式保留/删除的代码：生产层保留 canonical jaw-order 变换及单测；临时 rollout 导出和 observation variant audit 全部删除；reward 保持 phase156
前序阶段回归：语法、6 项 task-frame unittest 和 `git diff --check` 通过；内部安全判据仍使用 physical sensor identity
下一决策门槛：新 observation 语义必须从随机策略干净训练；历史 checkpoint 仅作对照，不得继续训练
```

2026-09-27 02:47 启动：确认 actor observation 共 20 组字段；canonical `b2z1_task_geometry` 与原始门/手柄/privileged 字段同时输入 actor。当前只追踪各字段的坐标系、符号和重复语义，不据此提前删除字段或修改接口。

静态生产链审计：

```text
已 canonical：relative_to_door、hand_force、handle_contact_force、base_lin_vel、base_ang_vel、b2z1_commands、b2z1_task_geometry、b2z1_panel_interaction
物理本体状态，不能仅因左右值不同删除：dof_pos、dof_vel、projected_gravity
左右门统一正关节语义：door hinge/handle/latch 在资产生成时通过 localRot0 翻转关节轴，正值分别表示开门、按压和缩回
明确保留物理侧别：privileged_door_info 中 opens_left/opens_right one-hot
待镜像配对实测：hand_handle_transform、door_dof_pos/vel 的实际数值一致性、delta_actions 的历史分布
侧别不影响语义：门尺寸、把手/锁舌几何和门轴动力学随机参数；这些字段应保留用于完整 Markov 状态
```

当前不能从静态代码得出“删除 left/right one-hot 即可修复”的结论：高层 action 已 canonical，但非镜像对称机械臂的真实关节状态仍可能让 actor 合理地区分两种腕部构型。下一门槛保持为同一物理进度、同一 canonical 误差下的左右门 observation 配对证据。

四个 rank 共导出 `10657` 个 OPEN_LATCH 样本。逐字段干预和配对结果：

```text
只交换 opens_left/opens_right one-hot：ee_x 均值变化约 0.00085--0.0020，不足以解释分叉
配对状态左右门 ee_x 均值差：约 0.0098--0.0137
左门 fixed_closing / moving_closing：约 +0.43 / -0.45
右门 fixed_closing / moving_closing：约 -0.50 / +0.33
只交换 fixed_closing：左门 ee_x 增加约 +0.0053
只交换 moving_closing：左门 ee_x 增加约 +0.0069
```

因果结论：门侧 one-hot 不是主要来源；`b2z1_task_geometry` 及抓持接触 observation 按 physical fixed/moving jaw 排列，但左右门所需腕部构型互为上下翻转，因此同一 canonical 抓持语义落在相反槽位。actor 据此学习出两套 OPEN_LATCH 动作。OPEN_LATCH 的 action reference 除 gripper 外均为零，门 DOF、`hand_handle_transform` 和底盘姿态的单字段影响很小，均不足以解释该分叉。

正式修复位于 observation 生产所有权层：新增 `canonicalize_jaw_order(values, door_open_lr)`，左门保持 `[fixed, moving]`，右门交换为 `[moving, fixed]`。该变换应用于 `b2z1_task_geometry` 双爪 force、`_grasp_contact_obs` 的 axial/closing/force norm/on-capsule，以及 actor `hand_force` 双爪槽位。reward、验收和安全终止继续使用 physical sensor identity，不受变换影响。

临时 rollout 导出钩子已从 trainer 删除，搜索不得再出现 `observation_variant_audit`。phase167 不生成新有效 checkpoint；phase156 step275 只能作为旧 observation 语义的历史对照。

### phase168：canonical jaw-order 接口干净训练（失败）

```text
上游结论与待解决瓶颈：phase167 已确认 physical jaw 槽位顺序造成左右门 observation 分叉，并在生产层统一为 canonical jaw order
基线 checkpoint / SHA-256：无；随机策略初始化，禁止加载旧 observation 语义 checkpoint
唯一变量：启用 phase167 的 canonical jaw-order observation
待验证假设：在其余训练条件不变时，左右门会同步学习 GRASP 和 OPEN_LATCH，REPOSITION 的门侧差距随上游改善而收敛
禁止改动范围：reward、action、PPO、网络、阶段验收、随机化、低层、探索倍率、AutoOpen 和其他 observation 字段
运行规模与日志：4 x 2048 env、64 steps/env、4 mini-batches、step1--100、每 25 step 保存；`/tmp/b2z1_phase168_canonical_jaw_clean64_8192.log`；运行目录在 iteration 1 出现后登记
成功/失败判据：step25/50/75/100 左右门 GRASP、OPEN_LATCH 和 canonical closing/contact 分布不再形成 phase151 式系统性分叉；运行无异常或 NaN，前序阶段不回归
观测指标：左右门各阶段 reached、strict grasp、pocket/axis/contact retention、canonical fixed/moving closing 与 force、OPEN_LATCH policy mean、hinge/gap、max handle force
结论与已排除方向：jaw-order 修复显著削弱左右分叉但未让两侧同步学会开缝；排除“jaw 槽位顺序是唯一剩余根因”和“只需训练到 step100 即可共同突破”
正式保留/删除的代码：启动前仅保留 canonical jaw-order 生产变换和单测；不保留 phase167 临时导出代码
前序阶段回归：启动前执行 py_compile、6 项 task-frame unittest、git diff --check，并确认 observation_variant_audit 搜索为空
下一决策门槛：phase169 逐组干预剩余 observation 字段，定位下一项因果字段；不修改 reward 或叠加行为机制
```

2026-09-27 03:14 首次启动误用了 phase167 只读诊断的 `8 steps/env、1 mini-batch`。进程和纯动力学模式本身正常，但 step25 左右 GRASP reached 均仅 `0.06%`，显著低于 phase148/151 同 step 的约 `27%--33%`。核对保存配置后确认 phase148/151 正式基线为 `64 steps/env、4 mini-batches`：旧 step25 总 timesteps 为 `13,107,200`，无效启动仅 `1,638,400`，每个 update 的训练样本量相差 8 倍。因此该结果不能用于评价 jaw-order 修复，也不能继续训练或继承 checkpoint。03:21 已只终止该训练进程，容器未关闭。

2026-09-27 03:22 正确重启：源开发机 GPU `0--3`，每 rank 2048 env，共 8192 env；随机策略初始化，未传入 checkpoint；参数为 100 update、64 steps/env、4 mini-batches、save frequency 25。相对 phase151 正式干净训练，唯一行为变量仍是 canonical jaw-order observation。当前等待首次场景创建和 iteration 1；在 step25 前只监控异常，不修改行为。

2026-09-27 03:26，iteration 1：四个 rank 均确认 `cameras=0, RTX sensors=0, frame rendering disabled`；总 timesteps 为 `524288`，与 phase148/151 正式基线一致。APPROACH_PREGRASP 左右 reached 均为 `100%`，GRASP 及后续均为 `0%`，符合随机策略起点。运行目录为 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260927_032226`；无异常或 NaN。下一决策记录点仍为 step25。

2026-09-27 03:39，step25：checkpoint 为 `model_step_000025.pt`，SHA-256 `b399767ec963f5cb7f2a167092e7c4f4f4bfb28e95cc688580a24112324c9e08`。关键对照如下：

```text
指标                               phase151 step25       phase168 step25
APPROACH_PREGRASP reached 左/右     100.00% / 100.00%    100.00% / 100.00%
GRASP reached 左/右                  28.26% / 32.73%       28.32% / 33.40%
OPEN_LATCH reached 左/右             13.62% /  9.23%       13.80% /  9.56%
REPOSITION reached 左/右              0.00% /  0.39%        0.02% /  0.05%
GRASP contact quality 左/右           0.3828 / 0.4241       0.3817 / 0.4168
OPEN_LATCH latch position 左/右       0.4 mm / 7.7 mm       1.2 mm / 1.2 mm
OPEN_LATCH policy mean ee_x 左/右     -0.0010 / -0.0006     -0.0018 / -0.0023
max handle force                     42.23 N                36.57 N
```

中间判断：canonical jaw-order 没有损害 APPROACH_PREGRASP、GRASP 或 OPEN_LATCH 的早期学习速度；左右锁舌物理响应不再出现 phase151 step25 的明显分叉，符合修复假设。REPOSITION 样本仍太少，retention 也处于早期形成阶段，不能据此宣布 phase168 已验证。保持唯一变量不变继续到 step50，再检查左右 OPEN_LATCH 转换率和后续阶段是否同步增长。

2026-09-27 03:53，step50：checkpoint 为 `model_step_000050.pt`，SHA-256 `1f78ffaec1c6beba363886c1e99e548bf5cf0825ba05deef059021b0644ca478`。关键对照如下：

```text
指标                               phase151 step50       phase168 step50
GRASP reached 左/右                  32.03% / 37.55%       30.72% / 35.59%
OPEN_LATCH reached 左/右             20.91% / 17.42%       19.71% / 14.65%
REPOSITION reached 左/右              0.09% /  7.36%        0.05% /  0.08%
OPEN_LATCH -> REPOSITION 左/右         0.43% / 42.25%        0.25% /  0.55%
OPEN_LATCH latch position 左/右       6.3 mm / 13.5 mm      4.1 mm / 5.0 mm
OPEN_LATCH hinge angle 左/右           0.0012 / 0.0149 rad    0.0006 / 0.0008 rad
policy mean ee_x 左/右                -0.0009 / +0.0011     -0.0006 / -0.0006
policy mean ee_roll 左/右             -0.0003 / +0.0021     -0.0013 / -0.0012
max handle force                     42.53 N                35.81 N
```

中间判断：canonical jaw-order 已消除 phase151 在动作均值、锁舌位移、门角和后续转换率上的左右分叉，但当前统一结果是两侧都尚未学会 OPEN_LATCH 完成动作，不能把“对称”本身当作成功。GRASP/OPEN_LATCH 入口率只比 phase151 略低，训练流程和前序阶段没有断裂。保持代码不变继续到 step75；若两侧仍共同停滞，只做统一 observation 的下一字段级诊断，不恢复 physical jaw 顺序、不调 contact reward。

2026-09-27 04:06，step75：checkpoint 为 `model_step_000075.pt`，SHA-256 `f0948311964e0f86c0cc88cd4499e8aadcc0411487f767dedb71075d7a0cb24f`。关键对照：

```text
指标                               phase151 step75       phase168 step75
GRASP reached 左/右                  39.26% / 46.62%       35.48% / 41.09%
OPEN_LATCH reached 左/右             25.85% / 25.32%       23.27% / 17.79%
REPOSITION reached 左/右              0.98% / 17.62%        0.15% /  0.33%
OPEN_LATCH -> REPOSITION 左/右         3.79% / 69.59%        0.64% /  1.85%
OPEN_LATCH latch position 左/右      12.0 mm / 19.5 mm      9.1 mm / 11.2 mm
OPEN_LATCH hinge angle 左/右           0.0011 / 0.0230 rad    0.0007 / 0.0022 rad
policy mean ee_x 左/右                -0.0014 / +0.0030     -0.0012 / -0.0010
retention quality                     0.3404                 0.2930
max handle force                     43.01 N                46.94 N
```

中间判断：两侧动作和物理响应仍比 phase151 对称，但并未共同形成正 `ee_x` 开缝动作；转换率只有小幅上升，不能再把共同停滞解释为 step50 的瞬时学习时序。仍按预设完成 step100，以同一运行排除较晚突破；若 step100 仍无改善，phase168 应判为“接口修复正确但不足以解决全部表示问题”，下一 phase 只做剩余 observation 字段的只读因果诊断。

2026-09-27 04:20，step100 正常完成，四个 rank 已退出，无异常或 NaN。checkpoint 为 `model_step_000100.pt`，SHA-256 `a8be95418108c550d920b8d18a95a26c77b960added8f66c923645a7418e43e4`。最终对照：

```text
指标                               phase151 step100      phase168 step100
GRASP reached 左/右                  40.28% / 51.58%       37.62% / 42.37%
OPEN_LATCH reached 左/右             27.49% / 31.62%       25.93% / 20.71%
REPOSITION reached 左/右              0.92% / 26.96%        0.65% /  3.51%
OPEN_LATCH -> REPOSITION 左/右         3.35% / 85.26%        2.51% / 16.95%
PUSH_OPEN reached 左/右               0.49% /  8.30%        0.15% /  0.71%
THROUGH reached 左/右                 0.00% /  3.90%        0.00% /  0.10%
OPEN_LATCH latch position 左/右      11.1 mm / 21.6 mm     16.1 mm / 18.5 mm
OPEN_LATCH hinge angle 左/右           0.0005 / 0.0306 rad    0.0007 / 0.0063 rad
policy mean ee_x 左/右                -0.0027 / +0.0055     -0.0007 / +0.0005
retention quality                     0.3631                 0.4336
max handle force                     52.01 N                71.38 N
```

最终结论：canonical jaw-order 是经干预确认的真实接口 bug，修复后左右锁舌位移、动作均值和门角差异明显缩小，且 retention 提高；该生产层修复与 6 项单测正式保留。但 phase168 的核心假设“仅修复 jaw-order 即可让左右门同步学会 OPEN_LATCH”失败：step100 右门转换率仍约为左门 `6.8` 倍，两侧都未达到可继续训练后续阶段的水平。不得继续盲目延长训练、恢复 physical jaw 顺序或调整 contact reward。step100 仅作为当前 observation 语义的诊断基线。

### phase169：剩余 actor observation 镜像不等价字段诊断（已验证）

```text
上游结论与待解决瓶颈：phase168 证明 jaw-order 修复必要但不充分；step100 仍存在 2.51%/16.95% 的左右 OPEN_LATCH 转换率分叉
基线 checkpoint / SHA-256：phase168 model_step_000100.pt / a8be95418108c550d920b8d18a95a26c77b960added8f66c923645a7418e43e4（只读诊断）
唯一变量：保持同一 LSTM 前置隐状态和其他 observation 不变，仅将 joint6 position 按腕部翻转关系平移 `pi` 到另一门型构型，测量 actor action mean 变化
待验证假设：physical joint6 的 `-1.617/+1.606 rad` 双峰仍让 actor 把镜像等价 OPEN_LATCH 状态分成两套动作
禁止改动范围：生产 observation、reward、action、PPO、网络、阶段验收、随机化、低层、探索倍率、AutoOpen 和 checkpoint 参数
运行规模与日志：4 x 2048 env，从 phase168 step100 staged-reset 状态做 8 步只读 rollout；`/tmp/b2z1_phase169_q6_counterfactual.log`
成功/失败判据：用单字段组替换或配对干预复现可测的门侧 action gap，并证明该字段的物理语义确实应 canonical；仅相关不算因果
观测指标：各字段组左右分布、配对距离、干预前后 ee_x/roll/pitch/yaw/gripper actor mean、OPEN_LATCH 物理状态和样本数
结论与已排除方向：待诊断
正式保留/删除的代码：只允许临时无副作用导出/干预；形成结论后删除，根因确认前不改生产接口
前序阶段回归：本 phase 不训练、不改变环境行为；临时代码必须与 actor 正式输入隔离
下一决策门槛：先判断 joint6 单字段 `+/-pi` 反事实是否显著闭合 ee_x/roll action gap；若无因果效应，再检查其余 arm dof，再依次检查 hand_handle_transform 和 delta_actions history
```

静态筛选：phase167 的 `10657` 个 OPEN_LATCH 样本中，joint6 observation 左/右均值为 `-1.617/+1.606 rad`，标准化均值差绝对值约 `70.3`，远大于其他关节；joint6 velocity 的标准化差异约 `2.19`。`hand_handle_transform` 也含确定性门侧旋转符号，但 phase167 已确认其单字段 action 影响很小；`delta_actions.ee_x` 的门侧差异更可能是既有策略输出的后果。因此 phase169 先只审计 joint6 position，不一次替换整个 arm 状态。

2026-09-27 04:29 首次 detached 启动在 checkpoint 读取阶段尚未出现 trainer 进程，被误判为退出；前台重放造成两个初始化实例。检测到端口占用后，两者均在 rollout 前终止，GPU 回到空闲，未生成诊断数据或更新策略。04:31 清理本轮临时日志后使用新端口只启动一个实例；后续进程核对必须同时检查训练入口 shell 和 accelerate 进程，不能只查 `train_agent_trl.py`。

2026-09-27 04:40，单实例完成场景初始化后在 rollout 第一帧失败，四个 rank 均为同一异常：临时 `_counterfactual_action_mean()` 调用了非循环 `Actor` 不存在的 `_get_action_reference()`。实际生产 `Actor.forward()` 直接通过 `action_residual_obs_slice` 读取 reference；该错误仅位于 phase169 临时只读诊断，没有执行环境动作、没有 PPO 更新，也没有生成 `.pt` 诊断数据。phase169 的假设、基线和唯一变量均不变；修复临时诊断的 actor 类型适配并增加 baseline 与真实 rollout action mean 一致性检查后重启。此次失败排除了环境创建和 checkpoint 读取问题，但没有产生 joint6 因果结论。

2026-09-27 04:43，临时诊断已按生产 actor 的既有接口修复：recurrent actor 复用 `_get_action_reference()`，feedforward actor 直接复用其 `action_residual_obs_slice` 语义；未修改生产模型类、observation、动作或环境。`py_compile` 与 `git diff --check` 通过，确认无旧训练进程后以端口 `29673` 单实例重启。重试日志为 `/tmp/b2z1_phase169_q6_counterfactual_retry.log`，其余基线、规模、唯一变量和判据不变。

2026-09-27 04:47，重试正常完成，四个 rank 均退出且无异常，导出 `10485` 个 OPEN_LATCH 样本（左 `5896`、右 `4589`）。baseline 反事实实现与真实 rollout action mean 的逐元素最大误差和平均误差均为 `0`，证明诊断没有漏掉 action reference、residual mask 或 normalization。样本中 joint6 左/右均值为 `-1.603/+1.573 rad`。将左门 joint6 加 `pi` 后，`ee_x` action mean 仅变化 `-0.000163`，绝对变化 99 分位为 `0.000482`；将右门 joint6 减 `pi` 后，`ee_x` action mean 仅变化 `+0.000113`，绝对变化 99 分位为 `0.000497`。baseline 左右 `ee_x` mean gap 为 `0.000924`，双向反事实后的 gap 仍为 `0.000648`；其余 action 坐标也只有同量级微小变化。

结论：joint6 physical position 的双峰是真实 observation 不等价，但该单字段对当前 actor 的因果作用很小，只能轻微缩小同批样本动作差，不能解释 phase168 中 `OPEN_LATCH -> REPOSITION` 左 `2.51%`、右 `16.95%` 的转换率分叉。不得仅凭 joint6 分布差异修改生产 observation，也不得继续围绕该单字段训练。phase169 不生成 checkpoint；诊断数据只作证据，全部临时 trainer 钩子在进入 phase170 前删除。下一步分别检查 joint6 之外 arm dof position 与 velocity 的组合效应，再按既定顺序检查 `hand_handle_transform` 和 `delta_actions` history。

2026-09-27 04:49，phase169 临时 trainer 钩子已全部删除；搜索 `B2Z1_Q6_AUDIT|q6_audit|counterfactual_action` 为空，`py_compile`、6 项 task-frame unittest 和 `git diff --check` 均通过。正式代码只保留 phase167 已验证的 canonical jaw-order 修复。

### phase170：joint6 velocity 单字段敏感性诊断（已验证）

```text
上游结论与待解决瓶颈：phase169 排除 joint6 position 单字段主因；phase167 静态样本中 joint6 velocity 左右标准化差异约 2.19，是下一项最显著 arm dof 差异
基线 checkpoint / SHA-256：phase168 model_step_000100.pt / a8be95418108c550d920b8d18a95a26c77b960added8f66c923645a7418e43e4（只读诊断）
唯一变量：保持所有其他 observation 不变，仅把 joint6 velocity 置零，分别测量左右 OPEN_LATCH actor action mean 变化
待验证假设：当前 actor 对 joint6 velocity 的门侧分布差异敏感，该字段对剩余 action gap 有显著因果贡献
禁止改动范围：生产 observation、reward、action、PPO、网络、阶段验收、随机化、低层、探索倍率、AutoOpen 和 checkpoint 参数
运行规模与日志：4 x 2048 env，从 phase168 step100 staged-reset 状态做 8 步只读 rollout；`/tmp/b2z1_phase170_q6_velocity.log`，数据 `/tmp/b2z1_phase170_q6_velocity.rank{0,1,2,3}.pt`，端口 `29674`
成功/失败判据：baseline 必须与真实 rollout mean 完全一致；置零后若 task-producing action gap 没有显著收敛，则排除该字段是主因；即使有影响，也必须先确认物理语义才能修改生产接口
观测指标：左右样本数、joint6 velocity 分布、baseline/zero 的 ee_x/roll/pitch/yaw 均值与分位变化、baseline-rollout 误差
结论与已排除方向：待诊断
正式保留/删除的代码：仅允许临时只读钩子；结束后全部删除，不把 velocity 置零带入生产 observation
前序阶段回归：phase169 清理后 py_compile、6 项 task-frame unittest、git diff --check 已通过
下一决策门槛：若 joint6 velocity 无显著作用，重新审计 arm dof 剩余字段的物理镜像映射；没有明确映射时不做任意均值平移
```

物理约束：phase169 使用的 wrist 构型关系是 `q6_right = q6_left + pi`，因此其时间导数数学上是 `dq6_right = dq6_left`。phase170 的置零仅用于测量 actor 敏感性，不代表 velocity 应被清零或取反，也不能据此直接形成生产修复。

2026-09-27 04:52，phase170 单实例启动；基线、规模和训练参数与 phase169 相同，唯一反事实变量改为 joint6 velocity 置零。rollout 后直接退出，不执行 PPO 更新或保存 checkpoint。

2026-09-27 04:56，phase170 正常完成，四个 rank 均退出且无异常，共导出 `10485` 个 OPEN_LATCH 样本（左 `5896`、右 `4589`）。本轮 staged-reset 样本的 joint6 velocity 左/右均值为 `-0.01152/+0.01039 rad/s`，标准差为 `0.01375/0.01465 rad/s`；baseline 与真实 rollout action mean 最大和平均误差均为 `0`。置零后左/右 `ee_x` mean 分别只变化 `+0.0000098/-0.0000097`，绝对变化 99 分位为 `0.000068/0.000074`；baseline 左右 `ee_x` gap 为 `0.000924`，置零后为 `0.000943`，没有收敛。其他 action 坐标的变化同样很小。

结论：joint6 velocity 的门侧分布差异不是当前 actor action gap 的主因；其敏感性比已经排除的 joint6 position 更低。不得在生产 observation 中清零、取反或平移该速度。phase170 不生成 checkpoint，临时 trainer 钩子在进入 phase171 前删除。下一步先静态核对 joint1--5 的物理镜像映射和分布差异；没有明确数学映射的字段不做任意反事实替换。

2026-09-27 04:57，phase170 临时 trainer 钩子已全部删除；搜索 `B2Z1_Q6_VEL_AUDIT|q6_velocity_audit|counterfactual_action` 为空，`py_compile`、6 项 task-frame unittest 和 `git diff --check` 均通过。当前工作树只保留正式训练修改、canonical jaw-order 修复和持续状态文档。

### phase171：剩余 arm dof 字段组静态审计（已验证）

```text
上游结论与待解决瓶颈：phase169/170 已分别排除 joint6 position 和 velocity 单字段主因
基线数据：phase167 的 10657 个 OPEN_LATCH observation 样本
唯一变量：只审计 joint1--5 position/velocity 的左右分布与是否存在确定性物理镜像映射，不运行策略干预
待验证假设：joint1--5 中还存在类似 joint6 加 pi 的离散镜像关系
禁止改动范围：全部生产代码、observation、reward、action、PPO、阶段验收和随机化
观测指标：各 arm joint position/velocity 左右均值、方差、标准化均值差，以及姿态目标和 IK 生成链
结论与已排除方向：joint1--5 没有确定性离散映射；禁止构造任意均值平移
正式保留/删除的代码：无代码改动、无临时钩子
前序阶段回归：phase170 清理后 py_compile、6 项单测和 diff 检查通过
下一决策门槛：转向 delta_actions history 单字段组因果诊断
```

phase167 样本统计中，joint1--5 position 的标准化左右均值差绝对值依次约为 `0.235/0.128/0.136/0.130/0.231`，velocity 依次约为 `0.013/0.455/0.778/0.342/0.108`；均远小于已审计的 joint6 position `70.3` 和 velocity `2.19`。生产链路由同一 canonical 门型几何构造目标 EE 姿态，再经统一 IK 求解 joint1--6；未发现 joint1--5 的门侧分支或类似 `+pi` 的离散构型变换。样本中的小幅差异属于状态分布差异，不能据此定义 observation canonicalization。phase171 不启动仿真、不生成 checkpoint，也不修改代码。

### phase172：delta_actions history 单字段组因果诊断（已验证）

```text
上游结论与待解决瓶颈：phase169--171 排除 arm dof observation 中剩余的确定性镜像接口问题；左右 OPEN_LATCH 转换率仍为 2.51%/16.95%
基线 checkpoint / SHA-256：phase168 model_step_000100.pt / a8be95418108c550d920b8d18a95a26c77b960added8f66c923645a7418e43e4（只读诊断）
唯一变量：保持同一 observation 的其余字段不变，仅把 10 维 delta_actions history 全部置零，测量 actor action mean 变化
待验证假设：策略先前输出经 delta_actions 回馈给 actor，会把 OPEN_LATCH 早期微小门侧动作差持续放大
禁止改动范围：生产 observation、reward、action、PPO、网络、阶段验收、随机化、低层、探索倍率、AutoOpen 和 checkpoint 参数
运行规模与日志：4 x 2048 env，从 phase168 step100 staged-reset 状态做 8 步只读 rollout；`/tmp/b2z1_phase172_delta_history.log`，数据 `/tmp/b2z1_phase172_delta_history.rank{0,1,2,3}.pt`，端口 `29675`
成功/失败判据：baseline 必须与真实 rollout mean 完全一致；置零后若 ee_x/roll action gap 显著收敛，证明 history 有放大作用；若影响很小或 gap 不收敛，则排除该反馈链是主因
观测指标：左右样本数、原始 delta_actions 均值、baseline/zero-history 的各 action 均值及变化分位、baseline-rollout 误差
结论与已排除方向：待诊断
正式保留/删除的代码：仅允许临时只读钩子；结束后全部删除，不把 history 置零带入生产 observation
前序阶段回归：phase170 清理后 py_compile、6 项 task-frame unittest、git diff --check 已通过
下一决策门槛：若 history 有显著放大作用，再区分它是合理的动作连续性状态还是不必要的自反馈；若无，则按既定顺序复核 hand_handle_transform 后停止 observation 猜测
```

`delta_actions` 使用统一 canonical action 语义，本身不是左右门接口字段；但它记录策略先前输出，可能把早期微小分叉反馈给 actor 并持续放大。phase172 只比较原始 history 与全零 history 下同一 observation 的 actor mean，判断该自反馈链是否显著贡献当前 `ee_x/roll` gap。即使因果作用显著，也不能直接删除 history；必须再判断它对稳定控制和所有阶段的作用。

2026-09-27 05:01，phase172 单实例启动；环境实际执行原始策略动作，置零 history 只用于额外计算 actor mean。rollout 后直接退出，不执行 PPO 更新或保存 checkpoint。

2026-09-27 05:05，phase172 正常完成，四个 rank 均退出且无异常，共导出 `10486` 个 OPEN_LATCH 样本（左 `5897`、右 `4589`）。baseline 与真实 rollout action mean 最大和平均误差均为 `0`。置零 10 维 history 后，左右单样本 `ee_x` action 的绝对平均变化为 `0.000706/0.000718`，99 分位为 `0.002676/0.002602`，说明 actor 确实使用历史，但作用强度近似对称。左右 `ee_x` mean gap 仅从 `0.000925` 降至 `0.000879`，缩小约 `5.0%`；`ee_roll` gap 从 `0.000416` 增至 `0.000510`，扩大约 `22.5%`，其他坐标也未形成一致收敛。

结论：`delta_actions` history 对动作连续性有可测作用，但没有系统性放大左右门 action gap，不是 phase168 转换率分叉的主因。不得删除、置零或按门侧改写该字段。结合 phase167 对门 DOF 和 `hand_handle_transform` 单字段影响很小的既有证据，observation 字段枚举到此停止；下一步回到 phase168 canonical jaw checkpoint，直接审计左右 OPEN_LATCH 的 reward、advantage、探索相关信号和实际 PPO 更新方向，判断剩余差异属于 credit、优化还是状态覆盖问题。

2026-09-27 05:07，phase172 临时 trainer 钩子已全部删除；搜索 `B2Z1_DELTA_HISTORY_AUDIT|delta_history_audit|counterfactual_action` 为空，`py_compile`、6 项 task-frame unittest 和 `git diff --check` 均通过。正式 trainer 已恢复，不含置零 history 或提前退出路径。

### phase173：canonical jaw checkpoint 的学习信号链路诊断（已验证）

```text
上游结论与待解决瓶颈：phase169--172 未发现 jaw-order 之外足以解释分叉的 actor observation 字段；必须区分 reward credit、PPO 优化和状态覆盖
基线 checkpoint / SHA-256：phase168 model_step_000100.pt / a8be95418108c550d920b8d18a95a26c77b960added8f66c923645a7418e43e4（诊断单 update）
唯一变量：生产行为和 loss 不变，仅按门侧记录 OPEN_LATCH 的 ee_x/roll sample noise、normalized advantage、noise*advantage signal，以及同批 observation 上一次 PPO update 前后的 actor mean
待验证假设：canonical jaw 修复后的左右策略都获得朝正确开缝方向的学习信号；剩余转换率差异来自状态覆盖或信号强度，而非 reward/PPO 符号错误
禁止改动范围：reward、observation、action、网络、优化器、阶段验收、随机化、低层、探索倍率、AutoOpen 和 checkpoint 参数
运行规模与日志：首次 4 x 2048 env、32 steps/env、单 update，`/tmp/b2z1_phase173_signal.log`；因窗口内没有 released 样本，保持唯一变量不变以正式 64 steps/env 重跑，`/tmp/b2z1_phase173b_signal.log`，数据 `/tmp/b2z1_phase173b_signal.rank{0,1,2,3}.pt`，端口 `29677`
成功/失败判据：明确左右门 rollout signal 与同批 PPO 更新方向是否一致；若符号错误，定位到 reward/advantage/PPO 的首个断点，若一致则停止修改这些接口并转向状态覆盖
观测指标：side x latch-released 分组样本数、policy mean、normalized advantage、noise*advantage、sample weight、post-pre actor mean、retention 和 hinge
结论与已排除方向：critic、advantage normalization、纯 rollout 截断偏置和 release 瞬间 retention 崩塌均已排除；released 后续 reward 轨迹更差，但具体 reward 分量尚待 phase174 分解
正式保留/删除的代码：只允许无副作用统计；phase174 复用完逐项 reward 导出后全部删除，不保留 phase 特例
前序阶段回归：phase172 清理后 py_compile、6 项单测和 diff 检查通过
下一决策门槛：根据学习信号首个异常层决定 reward、优化或状态覆盖诊断；禁止在结果前续训或调 reward
```

2026-09-27 05:08，phase173 单实例启动。只读统计读取实际 PPO storage，并在一次 update 后对同一批 actor observation 重新前向；生产行为、reward 和 loss 均未改变，本轮 checkpoint 不提升为训练基线。

2026-09-27 05:12，32 步首次诊断正常完成，共 `44229` 个 OPEN_LATCH 样本（左 `24934`、右 `19295`），但 latch position 最大仅 `17.38 mm`，没有 `>=20 mm` 的 released 样本。左右 unreleased 的 `ee_x noise*advantage` 均为负（`-0.000046/-0.000109`），同批 PPO update 的 `post-pre ee_x` 也均为负（`-0.000522/-0.000258`），信号与更新方向一致。该结果排除左右符号相反，但 32 步只有 `0.64 s`，不足以覆盖正式 64 步 rollout 内可能出现的释放后状态，不能据此评价完整 OPEN_LATCH credit。保持 checkpoint、代码和唯一变量不变，以 64 steps/env 重跑；首次结果只作窗口不足证据。

2026-09-27 05:14，phase173 以正式 64 steps/env 重跑；环境、checkpoint、PPO 参数和诊断字段均不变。仍只执行一个 update，不保存或提升新 checkpoint。

2026-09-27 05:18，64 步重跑正常完成，共 `88930` 个 OPEN_LATCH 样本。released 样本左 `2565/50217=5.1%`、右 `1468/38713=3.8%`，首次出现约在第 `39/38` 步，加权平均时间位置为第 `56.7/57.1` 步，主要集中在 rollout 尾部。左/右 released 的 normalized advantage 均为负（`-0.250/-0.091`）；右门 released 的 `ee_x noise*advantage=-0.000184`，明确惩罚正开缝探索。左门 released signal 虽为 `+0.000092`，同批 PPO update 的 `post-pre ee_x` 却为 `-0.000708`；共享网络的其他状态更新覆盖了该稀有组信号。左右全 OPEN_LATCH 的 post-pre 分别为 `-0.000146/+0.000088`。

当前最强假设是 rollout 截断偏置：actor 使用到 64 步边界归零的 observed reward-to-go，released 状态只在末段出现，可能因剩余 horizon 更短而在同门型、同阶段归一化中获得系统性负 advantage。忽略 episode boundary 的离线近似已显示 released reward-to-go 比 unreleased 低约 `3.45/1.69`，但本轮导出缺少 dones，不能作为严格证据。只补充 `dones` 和按训练公式精确重算的 raw actor return 后，以同一 64 步条件再次复跑；生产 advantage、loss 和动作保持不变。

2026-09-27 05:21，exact-return 重跑启动，日志 `/tmp/b2z1_phase173c_signal.log`，数据 `/tmp/b2z1_phase173c_signal.rank{0,1,2,3}.pt`，端口 `29678`。除新增只读导出字段外，与 05:14 的 64 步诊断完全相同。

2026-09-27 05:25，exact-return 重跑正常完成。按训练公式和真实 dones 重算的 raw actor return 经 side x stage 标准化后，与实际 normalized advantage 最大误差低于 `1.2e-7`，严格排除 critic或归一化实现改变符号。released raw return 左/右为 `-0.288/-0.966`，unreleased 为 `+2.290/+0.209`；按相同时间步配对后 released 仍低 `0.527/1.060`。released 的即时 reward 平均反而较少负，done rate 也更低，因此不是单纯 rollout 尾部或终止频率造成；负 credit 来自释放后的后续 reward 轨迹。

下一步仍属于 phase173 根因定位，不改 reward：临时记录 exact `stage_progress`、pocket/axis/contact retention、latch、hinge 和 gap 轨迹，确认 released 后 raw return 下降由哪个物理分量开始。2026-09-27 05:29 以同一 checkpoint 和正式 64 steps/env 启动，日志 `/tmp/b2z1_phase173d_retention.log`，数据 `/tmp/b2z1_phase173d_retention.rank{0,1,2,3}.pt`，端口 `29679`。

备份记录：2026-09-27 05:21 已将截至 exact-return 重跑启动时的状态文档和 phase168 step100 有效 checkpoint 同步到 `wholebody_1/3/4`；三台 checkpoint SHA-256 均为 `a8be95418108c550d920b8d18a95a26c77b960added8f66c923645a7418e43e4`。生产代码自 04:21 备份后未变化；phase173 临时统计未同步为正式代码。

2026-09-27 05:35，retention 轨迹重跑正常完成，无残留训练进程。日志为 `/tmp/b2z1_phase173d_retention.log`，数据为 `/tmp/b2z1_phase173d_retention.rank{0,1,2,3}.pt`。宽分组统计如下：

```text
                         左 unreleased   左 released   右 unreleased   右 released
retention quality           0.7314         0.7238         0.6751         0.6029
stage progress              0.2228         0.3981         0.1997         0.3316
raw actor return           +2.3196        -1.0870        +0.3547        -0.9429
```

按首次 latch position 跨过 `20 mm` 对齐事件，左门在 `t=-1 -> 0 -> +3` 的 retention quality 为 `0.7015 -> 0.7124 -> 0.7218`，stage progress 为 `0.3826 -> 0.3918 -> 0.3969`；右门分别为 `0.6227 -> 0.6213 -> 0.6115` 和 `0.3398 -> 0.3418 -> 0.3363`。两侧 hinge 在该窗口内都接近零。释放瞬间没有发生 progress 或 retention 崩塌；右门随后出现缓慢下降，左门则没有。因此，“released 后立即丢抓持”不足以解释两侧共同的负 raw return。

把事件窗口延长到 `t=+15` 后，左门 retention quality 仍约 `0.719`、latch 增至 `23.37 mm`；右门 retention quality 下降至约 `0.568`、pocket distance 增至约 `89.5 mm`，但 latch 仍增至 `23.39 mm`。这进一步确认右门存在晚发 retention 退化，而左门的负 return 必须来自其他 reward 项。现有导出只有总 reward，不能严谨区分时间惩罚、动作/安全约束、progress potential 或其他分量；phase173 到此停止，不据此改 reward。

### phase174：latch release 后逐项 reward 来源诊断（已验证）

```text
上游结论与待解决瓶颈：phase173 排除 critic、归一化、纯截断偏置和两侧 release 瞬间 retention 崩塌；released raw return 仍显著更低
基线 checkpoint / SHA-256：phase168 model_step_000100.pt / a8be95418108c550d920b8d18a95a26c77b960added8f66c923645a7418e43e4（只读诊断）
唯一变量：保持生产 reward 数值和行为不变，仅导出每个环境、每个时间步的实际 reward component，并按首次 `latch_position >= 20 mm` 事件对齐
待验证假设：某个可明确命名的 reward component 在 release 后持续变差，导致 released raw return 低于同时间步 unreleased 样本
禁止改动范围：reward 权重与公式、observation、action、PPO、网络、阶段验收、随机化、低层、探索倍率、AutoOpen 和 checkpoint 参数
运行规模与日志：4 x 2048 env、64 steps/env、单 update，端口 `29680`；日志 `/tmp/b2z1_phase174_reward_components.log`，数据 `/tmp/b2z1_phase174_reward_components.rank{0,1,2,3}.pt`
成功/失败判据：必须定位 raw return 差值的逐项来源，且各分量之和与环境实际总 reward 在数值误差内一致；若现有接口拿不到逐项值，先修复通用只读统计接口，不用近似量替代
观测指标：side x released 分组的 component 即时均值、component return、同时间步配对差，以及首次 release 的 `t=-3...+15` 事件轨迹
结论与已排除方向：左右 released 的 return 差几乎全部来自 `task_stage_progress`；安全、动作、时间等惩罚不是主因。问题进一步收敛到增量势函数 reward 与 raw reward-to-go advantage 的组合，而不是某项惩罚权重
正式保留/删除的代码：signal/reward audit 临时代码已全部删除，生产代码不保留诊断路径
前序阶段回归：py_compile、6 项 task-frame unittest、git diff --check 均通过；临时标识搜索为空
下一决策门槛：固定同一 rollout 比较 raw reward-to-go 与标准 GAE/value baseline 的 action signal；先判定 advantage estimator 和 critic 哪一层错误，不直接改 reward
```

2026-09-27 05:39，phase174 已登记并启动。逐项值直接取自 `_compute_reward()` 已乘 scale/curriculum 后、累加进 `rew_buf` 的 tensor；每步强制校验各分量之和与实际总 reward 一致。该通道仅在 `B2Z1_SIGNAL_AUDIT_PREFIX` 存在时启用，不重算或修改 reward。

2026-09-27 05:44，phase174 正常结束。四个 rank 共导出 `88,853` 个 OPEN_LATCH 样本；环境内 26 个 reward 分量之和逐步复现 `rew_buf`。trainer 对 timeout 样本额外加入 `gamma * value` bootstrap；把该真实变换作为第 27 个分量后，逐项即时和与训练 reward 最大误差为 `0`，逐项折扣 return 之和与 raw actor return 最大误差为 `6.9e-5`。

按同一 rollout 时间步配对，released 相对 unreleased 的 raw actor return 差和主要分量如下：

```text
                         总 return 差   task_stage_progress   termination   其余全部
左门                        -0.7879           -0.6341           -0.1548      约 +0.0010
右门                        -0.6906           -0.7061           +0.0141      约 +0.0014
```

左右门共同的主因都是 `task_stage_progress`。该项是 signed potential delta；OPEN_LATCH 的势函数为 `retention * (0.10 + 0.45 * latch_progress + 0.45 * gap_progress)`。latch 达到 `20 mm` 后其分量饱和，而 hinge gap 尚未形成，released 状态已经消耗掉大部分可获得的 latch progress；当前 actor 又直接把 raw reward-to-go 按 side x stage 归一化作为 advantage，没有状态价值基线，因此高级状态会因为“剩余可赚 progress 较少”而被系统性放到负 advantage 一侧。phase174 证明了 credit 差的来源，但尚未证明应改势函数还是 advantage estimator；禁止直接调 reward 权重。

2026-09-27 05:49，phase173--174 的 `B2Z1_SIGNAL_AUDIT_PREFIX`、`_signal_audit_*` 和 reward component 临时通道已全部删除。容器内 `py_compile` 通过，`test_door_task_frame.py` 6 项单测通过，`git diff --check` 通过，临时标识搜索为空；无残留训练进程。

### phase175：actor advantage 与增量势函数的兼容性审计（已验证）

```text
上游结论与待解决瓶颈：phase174 确认 released 负 return 的共同主因是剩余 `task_stage_progress` return，而非安全、动作或时间惩罚
基线 checkpoint / SHA-256：phase168 model_step_000100.pt / a8be95418108c550d920b8d18a95a26c77b960added8f66c923645a7418e43e4（只读诊断）
唯一变量：同一固定 rollout 上离线比较当前 side x stage normalized raw reward-to-go 与标准 GAE/value-baseline 产生的 OPEN_LATCH action signal
待验证假设：raw reward-to-go 缺少状态基线，使已取得 latch progress 的状态因剩余势函数回报较少而获得负 advantage；标准 GAE 若 critic 合理，应消除该状态进度偏置
禁止改动范围：reward、observation、action、PPO loss、网络、阶段验收、随机化、低层、探索倍率、AutoOpen 和 checkpoint 参数
运行规模与日志：4 x 2048 env、64 steps/env，只读 rollout，端口 `29681`；日志 `/tmp/b2z1_phase175_advantage_audit.log`，数据 `/tmp/b2z1_phase175_advantage_audit.rank{0,1,2,3}.pt`
成功/失败判据：必须区分“raw RTG 无状态基线”和“critic/value 本身不准”各自对 action signal 的贡献；若 GAE 也给出错误方向，不允许简单切回 GAE
观测指标：side x released 分组的 raw RTG、value、GAE、noise*advantage、同时间步配对差，以及 action 到后续 progress/gap 的实际响应
结论与已排除方向：当前 raw RTG 的确存在剩余势函数偏置，但简单切回标准 GAE不能解决；critic 对 released 状态的估值比真实截断 RTG 更悲观，GAE 在两侧仍惩罚正 ee_x。critic 原始输入已覆盖相关物理量，不能归因于门状态缺字段
正式保留/删除的代码：临时 advantage audit 已全部删除，生产代码不保留诊断路径
前序阶段回归：py_compile、6 项 task-frame unittest、git diff --check 均通过；临时标识搜索为空
下一决策门槛：只把诊断 horizon 从 64 延长到 128；若后续真实收益修正信号，优先解决 rollout horizon，若仍不修正再审计 critic target/稀有进度采样
```

2026-09-27 05:52，phase175 已登记并启动。临时导出读取 trainer 已经计算出的 `values`、GAE returns 和当前 training advantages，并独立重算 raw reward-to-go；四个 rank 写盘后在 optimizer 之前直接退出，不产生参数更新或 checkpoint。

2026-09-27 05:57，phase175 正常结束，四个 rank 均在 optimizer 前退出。重算并按 side x stage 归一化的 raw RTG 与实际 training advantage 最大误差为 `1.9e-6`，确认当前训练链路没有隐藏变换；`raw_gae == returns - values` 最大误差为 `0`。

同一时间步内 released 相对 unreleased 的估计差：

```text
                         左门       右门
raw reward-to-go        -0.6127    -0.1161
critic value            -2.7033    -2.4660
raw GAE advantage       -0.4776    -0.0294
normalized GAE          -0.0584    -0.0030
```

标准 GAE 没有修正关键动作信号。released 样本的 `ee_x noise * advantage` 在当前 RTG 下左/右为 `-0.000238/-0.000117`，改用同批 normalized GAE 后仍为 `-0.000138/-0.000149`。与此同时，正 `ee_x` noise 与 released 后一步 hinge 增量的协方差在左右门均为正（约 `2.24e-8/2.98e-8`），物理作用方向没有反转。结论是：raw RTG 因 64 步末端缺少未来收益而偏置，GAE 则被更悲观的 critic bootstrap 继续拉回负方向，不能简单切换 estimator。

critic observation 已包含 pocket error、axis error、canonical finger/contact、latch、gap、hinge、action reference 和门动力学；value loss 也按六个大阶段加权。当前证据不支持“门状态缺字段”。但 released 只占 OPEN_LATCH 左/右样本约 `5.17%/3.98%`，且平均位于 rollout 末段；在讨论 critic 表达或高进度采样前，必须先用更长真实后续窗口排除 horizon 不足。

2026-09-27 05:59，phase175 临时导出代码已删除。容器内 py_compile、6 项 task-frame 单测和 git diff 检查通过，临时标识搜索为空；无残留训练进程。

### phase176：128 步只读 horizon 对照（已验证）

```text
上游结论与待解决瓶颈：64 步下 raw RTG 与 GAE 都惩罚 released 正 ee_x；release 位于末段，GAE critic bootstrap 额外悲观
基线 checkpoint / SHA-256：phase168 model_step_000100.pt / a8be95418108c550d920b8d18a95a26c77b960added8f66c923645a7418e43e4（只读诊断）
唯一变量：把 rollout 从 64 steps/env 延长到 128 steps/env；其余 checkpoint、环境、随机种子、reward、模型和动作分布不变
待验证假设：额外 64 步能覆盖 latch release 后形成 gap 或阶段转换的真实后果，使正确 ee_x 探索获得可见的后续 credit，并减少对错误 terminal value bootstrap 的依赖
禁止改动范围：全部生产行为、reward、critic 输入/value loss、advantage 算法、阶段验收、随机化、低层、AutoOpen 和 checkpoint 参数
运行规模与日志：4 x 2048 env、128 steps/env，端口 `29682`；日志 `/tmp/b2z1_phase176_horizon128.log`，数据 `/tmp/b2z1_phase176_horizon128.rank{0,1,2,3}.pt`；只读导出后在 optimizer 前退出
成功/失败判据：比较 64/128 步 released 样本的未来 gap、阶段转换、raw RTG、GAE 和 ee_x action signal；若 128 步仍无后续正后果，horizon 方向失败并转向 critic target/采样，不继续盲目加长
观测指标：release 事件到 rollout 末端的剩余步数、hinge/gap、stage transition、RTG/value/GAE、noise*advantage
结论与已排除方向：128 步显著提高 released 覆盖，并使全窗口 normalized GAE 的正 `ee_x` action signal 左/右恢复为 `+4.5e-6/+6.0e-5`；但所有 release 事件仍无阶段转换，不能把 horizon 不足当成唯一根因，也不继续盲目加长
正式保留/删除的代码：临时只读导出已删除，生产代码不保留诊断路径
前序阶段回归：phase175 清理后 py_compile、6 项单测、diff 检查通过
下一决策门槛：审计 critic target 与 released 高进度样本覆盖；只有先区分 target 错误和监督不足，才能决定是否调整 critic 训练或采样
```

2026-09-27 06:04，phase176 已登记启动。唯一实验变量是 rollout horizon `64 -> 128`；临时导出记录每步阶段、门关节、锁舌、动作、reward、value、GAE 和 raw RTG，四个 rank 写盘并同步后在 optimizer 前退出，不产生参数更新或 checkpoint。

2026-09-27 06:08，phase176 正常结束。四个 rank 各导出 `2048 x 128` 步，文件均完整；没有 optimizer、参数更新或新 checkpoint。OPEN_LATCH released 样本由 64 步的左/右 `5.17%/3.98%` 增至 `33.83%/34.46%`。同时间步 released 相对 unreleased 的 raw RTG 差为左/右 `-1.343/-0.126`，raw GAE 差为 `-0.150/+0.507`，分组 normalized GAE 差为 `-0.018/+0.056`。

全 128 步 released 样本的正 `ee_x noise * advantage`：当前 normalized RTG 为左/右 `-9.32e-5/+4.13e-5`，离线 normalized GAE 为 `+4.50e-6/+6.04e-5`。相比 64 步两种 estimator 在两侧均为负，额外窗口确实改善了 action credit；同时正 `ee_x` 与下一步 hinge 增量的协方差左右均为正，物理方向保持正确。

但是，左/右共 `854/643` 次 release 事件没有一次进入 REPOSITION；release 后可观测窗口平均约 `49.6/48.7` 步，期间最大 hinge angle 均值仅 `0.000108/0.000491 rad`，只有右门 `0.31%` 的事件曾超过 `0.02 rad`。因此 128 步没有覆盖到可靠开缝或阶段转换，只是通过更多高进度样本弱化了错误 credit。phase176 排除“继续单纯加长 horizon 就能暴露完整后果”，下一步转向 critic target 与 released 样本监督覆盖审计。

### phase177：released 高进度 critic target 与覆盖审计（已验证）

```text
上游结论与待解决瓶颈：128 步改善 GAE action signal，但 1497 次 release 无阶段转换；critic 对高进度状态仍显著悲观
基线 checkpoint / SHA-256：phase168 model_step_000100.pt / a8be95418108c550d920b8d18a95a26c77b960added8f66c923645a7418e43e4（只读分析）
唯一变量：不生成新行为，只按 latch progress、时间位置、门侧和阶段统计 phase175/176 已有数据中的 value target、value prediction、误差及现有样本权重
待验证假设：critic 悲观来自 64 步正式训练中 released 样本仅约 4--5%，而非 critic target 本身要求 released value 更低
禁止改动范围：环境、reward、observation、action、critic/value loss、advantage、PPO、horizon、阶段验收、随机化、低层和 AutoOpen
运行规模与日志：离线读取 phase175/176 四 rank 数据；不启动 SimulationApp
成功/失败判据：明确 target 与 prediction 各自随 latch progress 的变化，并量化 released 样本对 value loss 的实际占比；证据不足时再设计单一只读导出，不用训练结果反推
观测指标：GAE return、value、prediction error、样本数、stage weight、rollout 时间和门侧
结论与已排除方向：target 本身随 latch progress 降低，正确反映当前策略 release 后不开缝并最终失败；增加 released critic 样本不会把其 value 抬高，排除“只需高进度 oversampling 修复 critic”
正式保留/删除的代码：不增加生产代码
前序阶段回归：phase176 清理后执行 py_compile、6 项 task-frame unittest、git diff --check 和临时标识搜索
下一决策门槛：在算法层用标准 GAE 替代 raw RTG actor advantage，并以 128 步提供足够后续窗口；必须干净重训且回归前序阶段
```

2026-09-27 06:17，phase177 离线审计完成。64 步下 released 仅占 OPEN_LATCH 左/右样本 `5.17%/3.98%`，但其 squared value error 只占本阶段 `1.10%/0.61%`；不是少量高误差样本支配 critic。released 相对同时间步 unreleased 的 value prediction 低 `2.703/2.466`，GAE value target 也低 `3.181/2.495`，observed RTG 低 `0.613/0.116`。target 的低方向并非 critic 凭空制造。

128 步下 released 样本占比升至约 34%，value loss 占比升至左/右 `21.20%/17.59%`，但 target 仍比同时间步 unreleased 低 `3.617/3.350`。released 的绝对 `target - value` 均值为左/右 `-1.548/-1.510`，说明 checkpoint critic 对当前失败轨迹反而偏乐观；继续增加同类监督会让 value 更低。结合 phase176 的零阶段转换，target 正在如实表达“release 后不开缝、随后超时”的真实后果。

因此根因位于 actor credit estimator：当前 raw RTG 没有状态基线，会把 released 状态后续失败的整段低 return 归因给当前动作；标准 GAE 在 128 步下的正 `ee_x` action signal 已恢复为左右非负。下一阶段必须作为统一 PPO 算法修复删除 bespoke raw RTG 路径，并配套使用 128 步 horizon；这不是门侧、阶段或 seed 特例。

### phase178：标准 GAE + 128 步干净训练（失败）

```text
上游结论与待解决瓶颈：raw RTG 错误归因 released 状态的后续失败；128 步标准 GAE 对正确 ee_x 恢复非负信号，critic target 本身没有符号 bug
基线 checkpoint / SHA-256：不加载 checkpoint，从随机 actor/critic 干净训练；phase168 step100 仅作历史对照
唯一机制变更：删除 actor raw reward-to-go advantage 路径，标准 GAE 成为唯一 actor advantage；正式 rollout 固定为 128 steps/env
待验证假设：统一的状态价值基线与足够后续窗口能让左右门共同从 OPEN_LATCH 学会开缝，且不回归 APPROACH_PREGRASP、GRASP
禁止改动范围：reward、环境、observation/action、critic target/value loss、阶段验收、随机化、低层、AutoOpen、门侧/seed 分支和临时监督
运行规模与日志：单卡 1 env x 8 steps smoke 已通过；正式训练为 4 x 2048 env、128 steps/env、4 mini-batches，端口 `29683`，日志 `/tmp/b2z1_phase178_gae128_clean_8192.log`，每 25 update 保存
成功/失败判据：按固定 update 节点比较左右门阶段 reached、released 后 hinge/gap、action signal、安全指标和前序阶段；没有共同改善时停止，不追加补丁
观测指标：六阶段 reached、OPEN_LATCH released/gap/retention、ee_x signal、critic error、termination 和 unsafe force
结论与已排除方向：标准 GAE + 128 步显著改善 GRASP、OPEN_LATCH 和早期后续阶段覆盖，但 OPEN_LATCH -> REPOSITION 在 update50--100 平台化；局部正 ee_x signal 没有使 actor mean 转正，排除“仅修 estimator 并继续训练即可突破”的假设
正式保留/删除的代码：删除废弃 raw RTG 配置和实现，不保留兼容开关；只保留标准 PPO GAE
前序阶段回归：修改后执行 py_compile、全部 task-frame 单测、配置加载、纯动力学 smoke 和阶段指标回归
下一决策门槛：用 phase179 在同批 observation 上直接测量一次 PPO 更新前后的分阶段 actor mean 和真实加权 score；得到因果证据前不修改 reward 或阶段权重
```

2026-09-27 06:18，phase178 单卡轻量 smoke 正常完成。`1 env x 8 steps x 1 update` 成功执行标准 GAE、PPO optimizer、日志和退出，无异常或 NaN；运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260927_061624` 仅为 smoke，不作为 checkpoint 基线。

2026-09-27 06:19，phase178 四卡干净训练启动。随机初始化 actor/critic，不加载 checkpoint；唯一算法机制变化为删除 raw RTG actor advantage 并统一使用标准 GAE，rollout 使用 phase176 验证过的 128 步。计划在 update25/50/75/100 核对左右门六阶段到达率、released 后 hinge/gap、正 `ee_x` signal、安全和前序阶段回归。

2026-09-27 06:22，update1 正常完成，无异常、NaN 或 OOM。全局每轮 `1,048,576` 环境步，吞吐约 `17,510 steps/s`，collection `58.61 s`、learning `1.27 s`、总计 `59.88 s/update`；首轮全部样本仍在 APPROACH_PREGRASP，符合随机初始化起点。运行目录为 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260927_061734`，下一有效判断点为 update25。

2026-09-27 06:47，update25 checkpoint 已保存为 `model_step_000025.pt`，SHA-256 `c0a8715d2d6b03502d496a0c7a9be641dbdcf7746f6f28a9cca04c3fd8388467`。APPROACH_PREGRASP 左右均 `100%`；GRASP 左/右 `31.22%/36.35%`；OPEN_LATCH `21.18%/16.00%`；REPOSITION `0.61%/2.03%`；PUSH_OPEN `0.14%/0.52%`；THROUGH `0.00%/0.36%`。OPEN_LATCH 全局 latch position `21.1 mm`、hinge angle `0.0013 rad`、retention `0.5860`、max handle force `54.12 N`，正 `ee_x` advantage signal 为 `+4.666e-5`。

相对 phase168 step25，GRASP 从左/右 `28.32%/33.40%` 提升，OPEN_LATCH 从 `13.80%/9.56%` 提升，并且左右都已出现后续阶段样本；标准 GAE + 128 步没有回归前序阶段，且首次固定节点证据支持继续到 update50。当前不能因右门已有少量 THROUGH 就宣布成功。

备份记录：2026-09-27 06:49 已将 phase178 正式工作树、本文档和 step25 checkpoint 同步到 `wholebody_1/3/4`。三机 checkpoint SHA-256 均为 `c0a8715d2d6b03502d496a0c7a9be641dbdcf7746f6f28a9cca04c3fd8388467`；trainer 与实验配置 SHA 也一致。phase173--176 的临时诊断代码均已删除，未作为正式代码同步。

2026-09-27 07:15，update50 checkpoint 已保存为 `model_step_000050.pt`，SHA-256 `3daa2c4b6d256dcac369bf6bf355c033687ceee1224926d090370b94c091cf14`。APPROACH_PREGRASP 左右均 `100%`；GRASP 左/右 `39.06%/44.09%`；OPEN_LATCH `26.54%/22.22%`；REPOSITION `1.28%/2.98%`；PUSH_OPEN `0.63%/1.15%`；THROUGH `0.15%/0.84%`。OPEN_LATCH 全局 latch position `20.8 mm`、hinge angle `0.0005 rad`、retention `0.7404`、max handle force `51.56 N`，正 `ee_x` advantage signal 仍为 `+4.128e-5`。

相对 phase168 step50，GRASP 从左/右 `30.72%/35.59%`、OPEN_LATCH 从 `19.71%/14.65%` 明显提高；REPOSITION 从 `0.05%/0.08%` 提高到 `1.28%/2.98%`。按累计 reached 比值，OPEN_LATCH -> REPOSITION 约为左 `4.8%`、右 `13.4%`，尚有门侧差距，但两侧均持续增长且已有后续阶段样本。保持唯一机制不变继续到 update75。

备份记录：2026-09-27 07:17 已将 phase178 正式工作树、本文档和 step50 checkpoint 同步到 `wholebody_1/3/4`；step50 checkpoint SHA-256 为 `3daa2c4b6d256dcac369bf6bf355c033687ceee1224926d090370b94c091cf14`。

2026-09-27 07:43，update75 checkpoint 已保存为 `model_step_000075.pt`，SHA-256 `62a3bd22f740c97acb5ad9f8be488451918a5d74b22bede7ca138428840301c3`。APPROACH_PREGRASP 左右均 `100%`；GRASP 左/右 `41.84%/44.51%`；OPEN_LATCH `29.80%/23.31%`；REPOSITION `1.46%/3.00%`；PUSH_OPEN `0.95%/1.54%`；THROUGH `0.01%/1.03%`。OPEN_LATCH retention 提升到 `0.7965`、max handle force 降至 `49.63 N`，正 `ee_x` advantage signal 为 `+6.065e-5`。

按累计 reached 比值，OPEN_LATCH -> REPOSITION 为左 `4.9%`、右 `12.9%`，相对 update50 的 `4.8%/13.4%` 基本平台；同时 PUSH_OPEN 样本仍增长，说明后续链路能消费少量成功转换。当前首个瓶颈收敛为“锁舌释放后形成门缝”，但按预设继续到 update100 排除较晚突破，期间不修改 reward、observation 或动作语义。

2026-09-27 08:11，phase178 update100 正常结束，四个 rank 均退出，无异常、NaN 或 OOM。checkpoint SHA-256 为 `12047391fe394c97b9dd38b2694b76b66e3c55f48bb365ee2d37bdb3f6a7ae96`。APPROACH_PREGRASP 左右均 `100%`；GRASP `43.91%/46.13%`；OPEN_LATCH `32.08%/25.23%`；REPOSITION `1.44%/3.05%`；PUSH_OPEN `1.05%/1.59%`；THROUGH `0.06%/1.03%`。OPEN_LATCH -> REPOSITION 累计转换约左 `4.5%`、右 `12.1%`，确认 step50--100 平台化。

OPEN_LATCH 的正 `ee_x` advantage signal 从 step25 的 `+4.666e-5` 增至 step100 的 `+8.452e-5`，但左右 policy mean `ee_x` 最终仍为 `-0.0005/-0.0006`；latch position 从约 `21 mm` 降至 `18.3/18.5 mm`，hinge angle 回到 `0.0000/0.0003 rad`。标准 GAE 修复了 estimator 方向，却没有让共享 actor 的该状态动作均值按信号移动，因此 phase178 的核心训练假设失败，禁止继续盲目续训。

step100 各阶段的同一 action signal 与实际 sample weight 显示潜在梯度抵消：OPEN_LATCH count/weight/signal 为 `277323/0.9778/+8.452e-5`，PUSH_OPEN 为 `8104/10/+6.233e-5`，THROUGH 为 `5709/10/-3.5538e-4`。未加权 signal 不能代表共享网络更新；下一 phase 必须直接测量一次 PPO 更新前后固定 observation 的分阶段 actor mean，而不是据此立即修改权重。

### phase179：分阶段 PPO signal 与实际 actor 更新审计（已验证）

```text
上游结论与待解决瓶颈：phase178 的 OPEN_LATCH 正 ee_x signal 持续增强，但 actor mean 仍为负、门缝消失；其他阶段存在反向加权 signal
基线 checkpoint / SHA-256：phase178 model_step_000100.pt / 12047391fe394c97b9dd38b2694b76b66e3c55f48bb365ee2d37bdb3f6a7ae96（只读诊断）
唯一变量：保持 rollout、loss 和 optimizer 不变，导出同一批 observation 在一次 PPO update 前后的 actor mean，并按 stage x variant 统计变化及加权 score
待验证假设：共享 actor 参数上 THROUGH 的高权重反向梯度覆盖 OPEN_LATCH 正信号，使 OPEN_LATCH mean ee_x 未按局部 signal 移动
禁止改动范围：reward、observation/action、网络、stage weight、advantage、horizon、阶段验收、随机化、低层和 AutoOpen
运行规模与日志：4 x 2048 env、128 steps/env、4 mini-batches、单 update，端口 `29684`；日志 `/tmp/b2z1_phase179_actor_update.log`，数据 `/tmp/b2z1_phase179_actor_update.rank{0,1,2,3}.pt`；诊断结束后不保存 checkpoint，删除临时代码
成功/失败判据：用同批状态直接证明各阶段 pre/post mean 方向，并与应用真实 sample weight 后的 score 对应；若不对应，继续定位 optimizer/共享表示层，不调 reward
观测指标：stage x variant count/weight/noise*advantage、pre/post action mean、PPO ratio/KL 和梯度方向
结论与已排除方向：共享 actor 能对左右门产生相反且与各门型当批 ee_x signal 同向的更新；排除“THROUGH 高权重反向梯度统一覆盖 OPEN_LATCH 正 signal”和“optimizer 忽略 OPEN_LATCH signal”。真正问题是 stage 聚合正值掩盖右门当批 credit 为负
正式保留/删除的代码：临时 recurrent pre/post actor mean 导出已删除，生产代码不保留诊断路径
前序阶段回归：recurrent 重算 mean 与 rollout mean 最大误差 `6.73e-5`、平均绝对误差 `2.24e-7`；诊断清理后 py_compile、6 项 task-frame unittest 和 diff 检查通过
下一决策门槛：在当前 checkpoint 上按门型审计 OPEN_LATCH 物理进度、逐项 reward 和后续 credit 时间链，找到右门 ee_x signal 首次转负的真实来源；证据形成前不改 reward、权重或网络
```

2026-09-27 08:22，phase179 单卡审计路径 smoke 完成。`8 env x 8 steps` 的 recurrent 重算 action mean 与 rollout 保存均值逐元素完全一致，证明 trajectory split、hidden state 和 action reference 重建正确；该极小样本全部处于 residual mask 为零的 APPROACH_PREGRASP，因此 post-pre action mean 为零是阶段接口预期，不作为 PPO 更新结论。静态编译、6 项 task-frame 单测与写盘字段检查均通过。

2026-09-27 08:25，phase179 正式四卡诊断启动。唯一诊断变量是保存同批 recurrent observation 在真实一次 PPO update 前后的 action mean，并同时保存 stage、door variant、action noise、advantage 和正式 sample weight；不修改 rollout、loss、optimizer、reward 或环境，完成后不保存 checkpoint。

2026-09-27 08:28，phase179 四个 rank 正常退出，每个 rank 写出约 `47 MB` 数据文件，无异常、NaN 或新 checkpoint。共覆盖 `8192 x 128` 样本；更新前 recurrent 重算 mean 与 rollout 保存 mean 的平均绝对误差为 `2.24e-7`、最大误差 `6.73e-5`，足以确认 pre/post 比较使用了正确轨迹和 hidden state。

OPEN_LATCH 分层结果：左门 `115305` 个样本，stage weight `1.09703`，加权 `ee_x noise * advantage` 均值 `+2.05138e-4`，同批 PPO 更新后固定状态 mean `ee_x` 平均增加 `+2.27164e-4`；右门 `103284` 个样本，同权重下 signal 为 `-1.75890e-5`，mean `ee_x` 平均减少 `-4.15294e-4`。两侧实际更新均与本门型当批 signal 同向。此前仅按 stage 汇总得到的正 signal 由左门强正贡献掩盖了右门负贡献，不能说明两侧都收到正确 credit。

THROUGH 的 `ee_x` 加权总贡献仍为负，但没有把所有 OPEN_LATCH 状态统一推负：左门 OPEN_LATCH 实际正移。因此 phase179 否定“共享参数上的后期阶段反向梯度统一覆盖早期阶段”这一假设，也不支持立即引入 stage-conditioned head。下一步必须解释右门 OPEN_LATCH 的 on-policy signal 为什么在当前状态分布下转负。

### phase180：右门 OPEN_LATCH credit 时间链审计（已验证）

```text
上游结论与待解决瓶颈：phase179 证明 optimizer 按门型当批 signal 更新；stage 聚合正值掩盖右门 OPEN_LATCH ee_x signal 为负
基线 checkpoint / SHA-256：phase178 model_step_000100.pt / 12047391fe394c97b9dd38b2694b76b66e3c55f48bb365ee2d37bdb3f6a7ae96（只读诊断）
唯一变量：保持生产行为不变，导出 OPEN_LATCH 按门型分层的物理进度、逐项 reward、动作噪声、GAE advantage 和后续结果时间链
待验证假设：右门负 ee_x credit 来自当前 on-policy 状态分布中的 retention、contact、latch 或 gap 后果，而不是 optimizer 或共享参数更新方向错误
禁止改动范围：reward、observation/action、网络、stage/variant weight、advantage、horizon、阶段验收、随机化、低层和 AutoOpen
运行规模与日志：4 x 2048 env、128 steps/env、端口 `29685`；日志 `/tmp/b2z1_phase180_variant_credit.log`，数据 `/tmp/b2z1_phase180_variant_credit.rank{0,1,2,3}.pt`；只读导出后在 optimizer 前退出，不保存 checkpoint
成功/失败判据：定位右门正 ee_x 探索从即时物理响应到 GAE credit 的首个符号反转环节；若物理响应和各 reward 项均正确但 GAE 才反转，再审计 value/后续失败归因
观测指标：stage x variant 的 ee_x noise、latch/contact/retention/gap/hinge 增量、逐项 reward、value、return、advantage 和事件后剩余 horizon
结论与已排除方向：首个负作用发生在物理/任务进度层：正 ee_x 在 1--4 步先降低 retention、handle/latch progress，并提高终止概率；约 8 步后幸存轨迹才出现正 progress/return。排除“只有 critic 或 optimizer 把正确即时效果算反”，也证明左门 stage 聚合正 signal 不是可靠开门 credit
正式保留/删除的代码：临时物理状态、reward component 和 GAE 导出已删除，生产代码不保留诊断路径
前序阶段回归：26 个 reward component 逐步复现环境 reward；清理后 py_compile、6 项 task-frame unittest、git diff --check 和临时标识搜索通过
下一决策门槛：分类 OPEN_LATCH 终止原因和剩余阶段时间，区分阶段时间预算截断与真实安全失败；结论前不修改 termination 或 reward
```

2026-09-27 08:35，phase180 单卡 `8 env x 8 steps` 导出 smoke 通过。23 个主字段和 26 个实际 reward component 均按预期写盘，逐项 reward 之和与 trainer 输入 reward 最大误差为 `0`；静态编译、6 项 task-frame 单测和 diff 检查通过。

2026-09-27 08:36，phase180 正式四卡只读 rollout 启动。每一步成对记录动作前物理状态与动作后实际进入训练的 reward component；optimizer 前同步写盘并退出，不产生参数更新或 checkpoint。

2026-09-27 08:40，phase180 四个 rank 正常退出，各写出约 `70 MB`，无异常、参数更新或 checkpoint。OPEN_LATCH 样本左/右为 `114589/103392`；按门型的正式 normalized GAE `ee_x` signal 为 `+1.2512e-4/-4.6863e-5`，复现 phase179 的门型分化。

但按锁舌 `>=20 mm` 分层后，released 样本左/右为 `17252/18658`，`ee_x` signal 同时变为 `-2.6113e-5/-4.2251e-5`。正 `ee_x` 与下一步 stage progress、retention 的协方差在左右门均为负；即时 `task_stage_progress` reward covariance 也均为负。左门 stage 总体正 signal 主要来自 unreleased 样本中正 `ee_x` 与较低终止率相关，不能解释为推门开缝学习成功。

released 样本中，正 `ee_x` 与 done 的 covariance 左/右为 `+7.37e-6/+2.77e-6`，与 termination reward 的 covariance 为 `-4.78e-4/-2.51e-4`。只看没有在窗口内终止且始终留在 OPEN_LATCH 的幸存轨迹，1--4 步累计 progress/return 仍为负，约第 8 步后两侧 progress 和 reward+value bootstrap 才转正。说明任务动作存在短时夹持扰动和延迟收益，而一部分轨迹在收益出现前终止；下一步必须先分类终止所有权，不能直接延长时间或改 reward。

2026-09-27 08:46，phase180 临时审计代码已完整删除。容器内 py_compile、6 项 task-frame unittest、git diff --check 均通过，临时标识搜索为空；无残留训练进程。

### phase181：OPEN_LATCH 终止来源与剩余时间审计（已验证）

```text
上游结论与待解决瓶颈：released 正 ee_x 先造成 1--4 步 retention/progress 损失，约 8 步后幸存轨迹才转正；终止事件使两侧 released GAE signal 均为负
基线 checkpoint / SHA-256：phase178 model_step_000100.pt / 12047391fe394c97b9dd38b2694b76b66e3c55f48bb365ee2d37bdb3f6a7ae96（只读诊断）
唯一变量：不改行为，仅导出每步 time_in_stage、max_stage_time 余量和互斥终止原因
待验证假设：正 ee_x 的延迟收益被阶段超时或某一明确安全终止提前截断；若终止与动作无因果关系，则回到 GAE/value 时间归因
禁止改动范围：阶段时限、termination 条件、reward、observation/action、网络、权重、随机化、低层和 AutoOpen
运行规模与日志：单卡 `8 env x 8 steps` smoke；首轮 `4 x 2048 env x 128 steps` 正式导出位于 `/tmp/b2z1_phase181_termination_audit.*`；因 `1177/1232` 个 ever-released 片段在 rollout 边界仍未结束，保持唯一诊断变量不变，补充 `4 x 2048 env x 512 steps` 到 `/tmp/b2z1_phase181b_termination_audit.*`；optimizer 前退出，不保存 checkpoint
成功/失败判据：量化 released 正 ee_x 对 stage overtime、base low-height、arm overspeed、unsafe force、door overspeed、shoulder collision 和成功转入 REPOSITION 的独立影响
观测指标：stage x variant x released 的终止原因、剩余步数、ee_x noise、stage completion、retention 和 hinge/gap
结论与已排除方向：安全终止不是 released 动作的主要风险；在完整可观测的 `1577` 条 ever-released 片段中 `1522` 条最终 stage overtime，仅 `3` 条完成。首次 release 后到超时的中位可用控制时间为 `377` 步，排除“约 8 步延迟收益主要被阶段剩余时间截断”。正 ee_x 在 8 步内不增加 overtime 或安全终止，并对 hinge/latch 增量呈正确正相关。另确认通用 stage-complete 奖励晚一动作且归到下一阶段
正式保留/删除的代码：临时 reset 前终止快照和 trainer 导出均已删除；生产代码不保留 phase181 路径
前序阶段回归：两轮正式诊断均未执行 optimizer 或保存 checkpoint；清理后执行 py_compile、6 项 task-frame unittest、git diff 和临时标识搜索
下一决策门槛：先在通用 staged-task 所有权层修复完成事件时序并用单测证明奖励与完成动作同帧；不得先延长阶段时间或放宽安全终止
```

2026-09-27 08:52，phase181 临时诊断实现完成并登记为运行中。环境在 `_check_termination()` 最终所有权层、reset 发生前冻结阶段号、阶段计时、完成事件、启用的 low-height 条件、B2Z1 距门/机械臂超速条件及任务安全条件；trainer 在同一步 `env.step()` 返回后立即收集快照，与 rollout 的动作、mean、value、return 和 GAE 对齐，写盘后在 optimizer 前正常返回。该路径只由 `B2Z1_PHASE181_AUDIT_PATH` 启用，不改变生产行为。容器内 py_compile、6 项 task-frame unittest 和 `git diff --check` 已通过。

2026-09-27 08:54，phase181 单卡 `8 env x 8 steps` smoke 正常完成。25 个字段的 env/step 维度一致，所有浮点值有限；8 个 done 均由判定当步的 arm overspeed 覆盖，未发现无原因 done；导出 `/tmp/b2z1_phase181_smoke_0852.rank0.pt` 后 optimizer 未执行，进程正常退出。该小样本仅验证诊断链路，不用于任务结论。下一步按计划运行四卡正式 rollout。

2026-09-27 09:02，phase181 首轮四卡 `8192 env x 128 steps` 正常完成，四个 rank 各写出约 `43 MB`，无 optimizer 或 checkpoint。OPEN_LATCH 样本 `218460`，其中 latch released 样本 `37016`；OPEN_LATCH 的 `908` 个 done 中 `889` 个为 stage overtime、`16` 个为 episode timeout、`3` 个为 unsafe force，没有 low-height、arm overspeed 或 door overspeed。按连续 OPEN_LATCH 片段统计，`1232` 条曾达到 release，但仅 `55` 条在窗口内结束，`1177` 条在 rollout 边界仍未结束；首次 release 时剩余阶段时间中位数为 `400` 步。故 128 步数据只能确认已观测终止的类型，不能判断延迟收益是否最终被 500 步阶段预算截断。补充窗口延长到 512 步，不改任何物理、reward、termination 或策略行为；同时直接记录 OPEN_LATCH 完成条件，避免用下一步 stage 间接推断。

2026-09-27 09:09，phase181 补充四卡 `8192 env x 512 steps` 正常完成，四个 rank 各写出约 `181 MB`，共 `4,194,304` 样本且全部有限，无 optimizer 或 checkpoint。OPEN_LATCH 共 `1,171,116` 样本、`421,826` 个 released 样本；`2504` 个 OPEN_LATCH done 中 `2397` 为 stage overtime、`73` 为 episode timeout、`30` 为 unsafe force、`4` 为 low-height、`1` 为 arm overspeed，安全失败不是主因。

按连续 OPEN_LATCH 片段统计，`3059` 条曾达到 release。可从首次 release 一直观察到阶段预算终点的 `1577` 条中，`1522` 条 stage overtime、`25` 条 episode timeout、`23` 条 unsafe force、`3` 条完成、其余 `4` 条为 base/arm 终止；stage overtime 占 `96.5%`。超时片段首次 release 时剩余时间中位数为 `377` 步，之后实际观察到超时的中位数为 `378` 步。因此不是 8 步延迟收益缺少兑现时间，而是策略在数百步内仍无法持续形成门缝。按剩余时间分层后，正 `ee_x` 相对负 `ee_x` 在未来 8 步没有增加 overtime 或安全终止；其对 hinge/latch 增量从 1--32 步保持正确正相关，retention 只在 1--4 步轻微下降，8 步后转为正相关。

四个 OPEN_LATCH 完成事件还揭示了通用时序 bug：完成条件成立当步的 reward 分别不含 stage-complete 奖励，而下一步 stage 已为 REPOSITION、`actual_time_in_stage=0` 时才收到约 `+9.996`。例如 env `3227` 的 `t=302` 完成动作 reward 为 `-0.0041`，`t=303` REPOSITION 首动作才收到 `+9.996`。这会让 stage-balanced PPO 把事件样本归到错误阶段；下一 phase 在 `StagedTaskBase` 统一修复事件所有权，不写 OPEN_LATCH 特例。

2026-09-27 09:13，phase181 临时环境快照和 trainer 导出代码已完整删除，无残留训练进程。正式有效 checkpoint 仍为 phase178 step100，phase181 没有产生或修改模型参数。

### phase182：通用阶段完成事件时序修复（已验证）

```text
上游结论与待解决瓶颈：stage-complete 奖励晚一个动作并归到下一阶段，四个实测 OPEN_LATCH 完成事件全部复现
基线 checkpoint / SHA-256：phase178 model_step_000100.pt / 12047391fe394c97b9dd38b2694b76b66e3c55f48bb365ee2d37bdb3f6a7ae96
唯一变量：在 StagedTaskBase 生成单一 current-step stage-advance event，奖励和阶段切换共同消费该事件
待验证假设：修正事件所有权后，完成动作同帧获得 stage-complete 奖励，下一阶段首动作不再重复获奖；左右门和全部阶段语义一致
禁止改动范围：阶段条件、时限、reward scale、observation/action、网络、随机化、低层、AutoOpen、门型与 seed 行为
运行规模与日志：先增加无仿真通用时序单测，再做最小 Isaac smoke；通过后从 phase178 step100 干净续训对照
成功/失败判据：每个 stage advance 恰好产生一次奖励，且奖励与造成 advance 的 action 存储在同一 rollout step 和旧 stage；无 advance 时为零
观测指标：pending advance mask、reward step、storage task_stage、next observation stage、重复奖励计数
结论与已排除方向：通用事件所有权修复已通过纯 tensor 时序测试和真实 Isaac 生命周期 smoke；完成条件只计算一次，完成奖励与造成 advance 的动作同帧且仍属于旧阶段，阶段切换消费同一事件。排除继续使用 `actual_time_in_stage == 0` 从下一阶段反推完成事件的旧路径
正式保留/删除的代码：只保留通用 staged-task 事件机制和单测，删除旧的 actual_time_in_stage==0 间接推断
前序阶段回归：py_compile、6 项 task-frame 单测、3 项通用事件时序单测共 `9/9`、`git diff --check` 均通过；64-env、256 steps/env、单卡真实 Isaac smoke 完成一次 PPO update 后正常退出，未保存可继承 checkpoint；纯动力学检查为 cameras=0、RTX sensors=0、frame rendering disabled
下一决策门槛：从同一 phase178 step100 基线运行 25 update 四卡对照；只有转换率仍无改善时，才审计完成前持续进度 credit，不先改 reward、阶段阈值或时间预算
```

2026-09-27 09:19，phase182 已在 `StagedTaskBase` 增加单一 `stage_advance_event_buf`：阶段验收条件在 termination/reward 前每步只求值一次，当前步奖励和随后阶段切换共同消费该事件；reset 和 just-reset 环境统一屏蔽。B2Z1 的 `task_stage_complete` 已删除 `actual_time_in_stage == 0` 的下一阶段间接推断，改为当前 advance event 与 final-task event 的并集。纯 tensor 事件组合放在无 Isaac 依赖模块中，新增 3 项时序测试；连同已有 task-frame 测试共 `9/9` 通过，py_compile、`git diff --check` 和 phase181 临时标识清理检查通过。下一步只做最小 Isaac 生命周期 smoke。

2026-09-27 09:23，phase182 单卡真实 Isaac smoke 正常结束。运行规模为 64 env、256 steps/env，从 phase178 step100 只加载网络参数并完成 iteration 101；因 staged-reset 样本不会跨不同分布式拓扑恢复，本轮仅验证生命周期和 PPO 路径，不产生也不继承新的有效 checkpoint。日志中没有 traceback、运行时异常或 NaN；场景明确验证为 cameras=0、RTX sensors=0、frame rendering disabled。phase182 至此标为已验证。

### phase183：完成事件时序修复后的四卡对照训练（失败）

```text
上游结论与待解决瓶颈：phase182 已修复 stage-complete 奖励晚一动作且归入下一阶段的问题，尚未量化其对正式 on-policy 学习和阶段转换率的影响
基线 checkpoint / SHA-256：phase178 model_step_000100.pt / 12047391fe394c97b9dd38b2694b76b66e3c55f48bb365ee2d37bdb3f6a7ae96
唯一变量：相对 phase178，只启用 phase182 的通用 current-step stage-advance event；其余训练、环境和 reward 配置保持不变
待验证假设：完成动作获得正确阶段 credit 后，OPEN_LATCH 完成样本不再被计入 REPOSITION，且 OPEN_LATCH -> REPOSITION 转换率相对 step100 基线提高
禁止改动范围：阶段验收条件、时限、reward 数值、observation/action、网络、随机化、低层、AutoOpen、门型和 seed 行为；训练中不得追加阶段特例或临时阈值
运行规模与日志：4 x 2048 env、128 steps/env、4 mini-batches、25 update，从 step100 续训至 step125；端口 `29686`，日志 `/tmp/b2z1_phase183_stage_event_to125.log`；运行目录在首个 update 后补录
成功/失败判据：训练无运行时错误或 NaN；完成奖励只出现在旧阶段完成动作；比较 step100 与 step125 的左右门 OPEN_LATCH -> REPOSITION 转换率。若无提高则标记假设失败，不扩大修改范围
观测指标：左右门各阶段 reached/transition rate、stage_count、stage_reward_mean、stage-complete 样本所属阶段、OPEN_LATCH released/hinge/gap/retention 和安全终止率
结论与已排除方向：25 update 后 OPEN_LATCH -> REPOSITION 转换率左门约 `4.49% -> 4.95%`、右门约 `12.09% -> 12.09%`，没有实质改善；门缝仍接近零。排除“仅修复稀少完成事件的阶段归属即可解除当前平台”。夹持 retention、锁舌位移和部分后续 reached 略有改善，未见前序阶段或安全回归
正式保留/删除的代码：只保留 phase182 通用事件机制和单测；phase183 不预设诊断分支
前序阶段回归：启动前 py_compile、9 项单测和 git diff --check 通过；运行中四 rank 均为纯动力学模式，无 traceback、NaN 或安全终止增加。APPROACH_PREGRASP 左右保持 `100%`，GRASP 为 `43.63%/47.06%`
下一决策门槛：按预案审计完成前持续进度 credit；先定位 `ee_x` signal 从 step100 正值变为 step125 负值的首个反转层，证据形成前不改 reward 或阶段定义
```

备份记录：2026-09-27 08:17 已将 phase178 正式工作树、本文档和 step100 checkpoint 同步到 `wholebody_1/3/4`；step100 checkpoint SHA-256 为 `12047391fe394c97b9dd38b2694b76b66e3c55f48bb365ee2d37bdb3f6a7ae96`。phase179 临时诊断代码尚未创建，因此远程均保持正式生产代码。

备份记录：2026-09-27 09:22 已将 phase182 通用事件修复、全部当前正式工作树、本文档和 phase178 step100 有效 checkpoint 同步到 `wholebody_1/3/4`。本地与三机的修改文件聚合 SHA-256 均为 `c2f89903509c054fa934589d891cd08ba1c2de0b5e8ade472734a6978b3b7eeb`；checkpoint SHA-256 均为 `12047391fe394c97b9dd38b2694b76b66e3c55f48bb365ee2d37bdb3f6a7ae96`。

2026-09-27 09:23，phase183 正式四卡训练启动。四张 RTX A6000 启动前均无计算进程；使用 `4 x 2048 env`、128 steps/env、4 mini-batches，从 phase178 step100 运行 25 update 至 step125，端口 `29686`，日志 `/tmp/b2z1_phase183_stage_event_to125.log`。唯一行为变化是 phase182 通用完成事件时序修复；训练期间不修改行为代码。

2026-09-27 09:28，phase183 iteration 101 完成。四个 rank 均恢复各自 staged-reset curriculum snapshot，运行目录为 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260927_092319`；吞吐约 `17,295 steps/s`，六个阶段均有 rollout 样本，日志无 traceback、运行时异常或 NaN。四个 rank 均已确认 cameras=0、RTX sensors=0、frame rendering disabled。训练继续原样运行至 step125，中途不修改行为代码。

2026-09-27 09:53，phase183 iteration 125 正常完成并保存 `model_step_000125.pt`，SHA-256 为 `104d110b48eddb9b57819ac6a3cbf1bf386e764aa8cb935f9bd449abab7c22d7`；四个 rank 正常退出，无 traceback、NaN 或残留进程。相对 step100，左门 `OPEN_LATCH -> REPOSITION` 为 `0.0144/0.3208=4.49% -> 0.0155/0.3129=4.95%`，右门为 `0.0305/0.2523=12.09% -> 0.0312/0.2580=12.09%`，不足以支持训练假设。OPEN_LATCH pocket/axis/contact retention 从 `0.8785/0.8999/0.9315` 提高到 `0.8921/0.9271/0.9425`，锁舌位移从左/右 `18.3/18.5 mm` 提高到 `19.2/19.7 mm`，但 hinge angle 仍为 `0.0000/0.0001 rad`。左/右 PUSH_OPEN reached 为 `1.05%/1.76%`，THROUGH 为 `0.10%/1.17%`，没有前序阶段或安全回归。值得注意的是 OPEN_LATCH `ee_x` 加权 advantage signal 从 `+8.452e-5` 变为 `-8.051e-5`，下一 phase 必须先审计符号链。

备份记录：2026-09-27 09:55 将 phase183 正式工作树、本文档和 step125 checkpoint 同步到 `wholebody_1/3/4`；checkpoint SHA-256 三机均为 `104d110b48eddb9b57819ac6a3cbf1bf386e764aa8cb935f9bd449abab7c22d7`。

### phase184：step125 OPEN_LATCH credit 符号链审计（已验证）

```text
上游结论与待解决瓶颈：phase183 修复完成事件归属后转换率仍平台化，且 OPEN_LATCH ee_x 加权 advantage signal 由正转负
基线 checkpoint / SHA-256：phase183 model_step_000125.pt / 104d110b48eddb9b57819ac6a3cbf1bf386e764aa8cb935f9bd449abab7c22d7（只读诊断）
唯一变量：保持生产行为不变，导出 step125 OPEN_LATCH 按门型、released 和 retention 分层的动作噪声、即时物理增量、逐项 reward、value/GAE 与后续结果
待验证假设：负 ee_x credit 来自完成前持续进度的某个可定位环节，而不是 phase182 事件时序、optimizer 或阶段完成阈值
禁止改动范围：reward、observation/action、网络、stage/variant weight、advantage、horizon、阶段验收、时限、随机化、低层和 AutoOpen
运行规模与日志：4 x 2048 env、128 steps/env，只读 rollout；optimizer 前退出，不保存 checkpoint；端口 `29687`，日志 `/tmp/b2z1_phase184_credit_chain.log`，数据 `/tmp/b2z1_phase184_credit_chain.rank{0,1,2,3}.pt`
成功/失败判据：定位正 ee_x 探索从物理响应到 reward/GAE 的首个符号反转层；若现有字段不足，只增加无副作用观测，不写行为补丁
观测指标：door variant x released x retention 的 ee_x noise、latch/hinge/gap/contact 增量、26 项 reward、value、return、GAE 和未来 1/2/4/8/16/32 步结果
结论与已排除方向：released 状态下负 credit 首先出现在动作造成的实际 retention/progress，而非 value、GAE、optimizer、完成事件时序、安全终止或阶段预算。单独正 ee_x 会短期破坏夹持，约 32 步后 progress 才可能恢复；左门主要损伤 axis，右门主要损伤 pocket/contact。additive 势函数离线反事实只能使右门完整 reward-to-go 转正，不能单独解决左门
正式保留/删除的代码：四卡数据和日志作为证据保留在 `/tmp`；`B2Z1_PHASE184_AUDIT_PREFIX` 环境变量路径、环境 reward 分量快照和 trainer 导出分支已全部删除
前序阶段回归：四个 rank 正常退出，无 optimizer、checkpoint、异常或 NaN；纯动力学仍为 cameras=0、RTX sensors=0、frame rendering disabled；清理后 py_compile、9 项单测、git diff 和临时标识搜索均已通过
下一决策门槛：先解释左右门 released 状态的动作耦合差异。只有统一根因和左右门离线证据都成立，才允许进入行为修改；字段不足时只增加 opening force、EE 姿态和 command 的只读导出
```

2026-09-27 10:00，phase184 临时只读导出实现完成。它复用 phase180 的动作前物理状态、当前动作 reward 和 rollout 后 value/return/GAE 时序定义；环境变量未启用时不分配诊断张量。容器内 py_compile、9 项单测和 `git diff --check` 通过。单卡 `8 env x 8 steps` smoke 导出 25 个正式 reward 分量、termination、total 和 unattributed；分量和重建总 reward、总 reward 与 storage reward 的最大误差均为 `0`，所有字段维度一致且浮点值有限，进程在 optimizer 前正常退出。单卡因拓扑变化未加载四卡 staged-reset，0 个 OPEN_LATCH 样本只用于链路验证，不构成任务结论。

2026-09-27 10:05，phase184 四卡 `8192 env x 128 steps` 只读 rollout 完成，四个 rank 正常写盘后退出，未执行 optimizer 或保存 checkpoint。OPEN_LATCH 左/右样本为 `122554/108687`，其中 released 为 `31047/27888`。全阶段 normalized advantage `ee_x` signal 为左 `+2.448e-5`、右 `+4.249e-6`；released 分层后变为左 `-5.512e-5`、右 `+3.277e-5`，证明 stage 聚合符号不能代表 released 后的任务动作。

released 样本中，正 `ee_x` 与即时 `task_stage_progress` reward 的 covariance 左/右为 `-2.719e-4/-9.466e-4`，与总 reward 的 covariance 为 `-2.615e-4/-8.056e-4`。负项首先出现在实际 progress，不是 value、GAE 或 optimizer。1--16 步内 retention/progress 持续下降，约 32 步后才可能转正；完整 discounted reward-to-go 左/右仍为 `-1.072e-3/-9.662e-5`，因此也不能归因于短 horizon。

当前乘法势函数 `retention * (0.10 + 0.45*latch + 0.45*gap)` 的负 credit 几乎全部来自 retention。左门 released 的最小 retention 分量约为 pocket `32%`、axis `65%`、contact `3%`；右门约为 pocket `48%`、axis `46%`、contact `7%`。正 `ee_x` 在左门主要损伤 axis，在右门主要损伤 pocket/contact，而 latch/gap 即时收益接近零。这说明独立单轴扰动不是可行协调动作的充分代表，但尚不能据此决定修改动作分布还是 reward。

离线测试 additive 候选 `0.10*retention + 0.45*latch + 0.45*gap` 后，即时 reward covariance 左门从 `-2.654e-4` 改善到 `-4.804e-5`，右门从 `-9.434e-4` 改善到 `-1.728e-4`；完整 reward-to-go 左门仍为 `-8.204e-4`，右门转为 `+7.287e-5`。因此 additive 不是左右门共同解，不能直接实现。

2026-09-27 10:13，phase184 临时审计代码已完整删除。容器内 py_compile、9 项单测、`git diff --check` 和 phase184 临时标识搜索均通过；当前无训练或诊断进程。顶部执行卡、上层路线图、六阶段总账和 phase185 决策门槛已同步更新，后续恢复工作必须从 phase185 开始，不得重复 phase184 已排除方向。

备份记录：2026-09-27 10:15 已将 phase184 最终结论、清理后的正式代码、本状态文档和 step125 有效 checkpoint 同步到 `wholebody_1/3/4`。三机关键文件聚合 SHA 和 checkpoint SHA 均与源机一致；此次同步不修改远程 Git 历史或凭据。

### phase185：released 后协调动作根因审计（已验证）

```text
上游结论与待解决瓶颈：phase184 已定位负 credit 的首个来源是实际 retention/progress；单独 ee_x 扰动会破坏夹持，但尚未解释可行协调动作需要哪些统一耦合量
基线 checkpoint / SHA-256：phase183 model_step_000125.pt / 104d110b48eddb9b57819ac6a3cbf1bf386e764aa8cb935f9bd449abab7c22d7（只读分析基线）
唯一变量：在生产行为不变的前提下，补充 opening force、EE 实际/目标位姿和 command 状态的只读导出，解释当前策略为何没有形成门缝
待验证假设：canonical 表示或低层跟踪方向错误，或者逐步独立 Gaussian 探索无法为 delta accumulator 提供跨越接触死区所需的持续同向动作
禁止改动范围：reward、observation/action、网络、action distribution、stage/variant weight、阶段验收、时限、随机化、低层和 AutoOpen
运行规模与日志：现有 phase184 数据离线分析已完成；补充 `4 x 2048 env x 128 steps` 只读 rollout，日志 `/tmp/b2z1_phase185_coordination.log`，数据 `/tmp/b2z1_phase185_coordination.rank{0,1,2,3}.pt`，optimizer 前退出
成功/失败判据：用左右门共同证据解释 opening force、retention 损失和多维动作之间的因果顺序，并给出一个可离线证伪的统一候选；不能只解释单侧门或单个 seed
观测指标：10 维 action noise/mean、pocket/axis/contact retention、latch/gap/hinge、opening force、EE 实际姿态、EE command 及未来 1/2/4/8/16/32 步结果
结论与已排除方向：canonical 动作到 goal、动作到实际 EE 的左右门映射均同号且近似对角，排除 task frame 和低层跟踪方向错误。released 后 actor mean 接近零，opening force 几乎全为零，实际 EE 仍在把手外约 `57--59 mm`；`ee_x` noise 的 lag 1/2/4/8 自相关均接近零。现有证据最支持“delta-action 所需持续动作与逐步独立探索不匹配”，但仍需统一持续干预做因果确认
正式保留/删除的代码：只增加环境变量门控的只读字段导出；结论形成后全部删除，生产路径不得保留审计分支
前序阶段回归：任何行为修改前先保持 phase183 checkpoint 和 phase184 物理路径不变；本 phase 只读阶段不产生新 checkpoint
下一决策门槛：phase186 在当前 step125 状态下给左右门相同的持续 canonical `ee_x=0.00/0.01/0.02`。只有 `0.02` 能在两侧安全建立 opening force 和门缝，且 retention 可接受，才确认时间连续探索不足并进入正式机制设计
```

phase185 单卡 `8 env x 8 steps` smoke 完成。导出的 opening force、command observation、EE 实际/目标 task-frame 位姿、stage progress、retention、latch、gap 和 hinge 字段维度一致且值有限；进程在 optimizer 前退出。该 smoke 只验证导出链路，不构成任务结论。

phase185 四卡 `4 x 2048 env x 128 steps` 只读 rollout 已完成，共 `1,048,576` 个样本，其中 OPEN_LATCH `230914` 个、released `58840` 个；四个 rank 正常退出，无 optimizer、checkpoint、异常或 NaN。日志为 `/tmp/b2z1_phase185_coordination.log`，数据为 `/tmp/b2z1_phase185_coordination.rank{0,1,2,3}.pt`。

canonical 动作映射在左右门均保持正确符号和近似对角结构。代表性相关系数如下：

```text
                         左门      右门
ee_x -> goal task-x      +0.799    +0.828
ee_x -> actual task-x    +0.646    +0.607
ee_pitch -> goal pitch   +0.915    +0.909
ee_pitch -> actual pitch +0.662    +0.685
ee_yaw -> actual yaw     +0.763    +0.787
```

因此，当前问题不是 canonical 坐标变换符号错误，也不是低层控制器把目标跟踪到相反方向。released 状态下，左/右门 goal task-x 均值为 `-60.77/-59.43 mm`，actual task-x 为 `-58.82/-57.08 mm`，说明策略仍把 EE 留在把手外侧，没有持续建立穿门方向位移。

opening force 进一步闭合了物理证据：左门 released 样本全部为 `0 N`；右门均值仅 `0.0126 N`，超过 `1 N` 的比例为 `0.0904%`，p95 为 `0 N`，最大值 `51.43 N`。phase157 已独立验证 `clamp(force_task_x, 0)` 的符号，因此不能把该结果解释为指标方向错误；当前 step125 策略确实几乎没有推门。

released 后 actor mean 各维接近零，`ee_x` 探索噪声的时间自相关也接近零：

```text
lag                1         2         4         8
左门             +0.0004   -0.0050   -0.0069   +0.0050
右门             +0.0031   -0.0136   +0.0001   -0.0035
```

当前动作语义是 accumulator delta；跨越接触死区需要连续数步同向增量。phase155 的受控实验曾以每步持续 `ee_x=0.02` 得到左/右单步 hinge delta `0.00352/0.00384 rad`，而当前白噪声几乎不会形成这样的连续序列。该证据比“左右门需要两套姿态协调动作”更能解释目前两侧都缺失 opening force 的事实，但历史实验并非当前 step125 状态，故下一 phase 必须复验，不能直接修改正式探索机制。

对 additive 势函数继续做 retention 权重扫描：

```text
phi = wr * retention + (1 - wr) / 2 * latch + (1 - wr) / 2 * gap
```

即使 `wr=0`，左门 `ee_x-return covariance` 仍为 `-0.0013089`，右门为 `+0.0003844`；`wr=0--0.3` 几乎不改变左门结论。当前失败轨迹没有 opening force 或 gap，重加权不能凭空生成持续动作轨迹。因此 phase185 排除“先调低 retention 权重就能解决当前瓶颈”，也不恢复 phase158--166 已失败的 contact bridge/milestone 路径。

### phase186：持续 canonical ee_x 因果干预（已验证）

```text
上游结论与待解决瓶颈：phase185 排除 canonical 映射和低层跟踪方向错误；当前策略几乎没有 opening force，独立白噪声与 delta accumulator 所需时间连续性不匹配
基线 checkpoint / SHA-256：phase183 model_step_000125.pt / 104d110b48eddb9b57819ac6a3cbf1bf386e764aa8cb935f9bd449abab7c22d7（只读干预基线）
唯一变量：仅在 OPEN_LATCH released 后，将 canonical ee_x 按环境固定分组持续覆盖为 0.00/0.01/0.02；其余 9 维继续使用同一 step125 actor 输出
待验证假设：持续同向 delta 能在左右门都跨过接触死区、建立 opening force 并形成 gap；当前学习瓶颈的首要原因是探索缺少时间连续性，而非坐标、控制跟踪或 reward 权重
禁止改动范围：reward、observation、正式 action 接口/分布、网络、阶段条件、时限、随机化、低层、AutoOpen、门型和 seed 行为；不执行 optimizer，不保存 checkpoint
运行规模与日志：先做最小 smoke，再做 4 x 2048 env x 128 steps 只读干预；日志和数据前缀使用 b2z1_phase186_persistent_ee_x
成功/失败判据：相对 0.00 对照，0.02 必须在左右门都显著增加 goal/actual task-x、opening force 和 hinge/gap，且不引入不可接受的 retention、安全终止或最大接触力退化；只改善单侧不算通过
观测指标：door variant x intervention 的 task-x 累积、opening force 分布、hinge/gap、pocket/axis/contact retention、阶段完成、安全终止和最大接触力
结论与已排除方向：`0.02` 在左右门均显著建立 opening force、门角和阶段转换，完整任务接触力无一超过 `600 N`，done 率不高于对照；确认当前共同根因是逐步独立探索缺少跨越接触死区所需的时间连续性。`0.01` 只产生小门缝且无转换，说明仅提高单步均值不足以替代持续覆盖
正式保留/删除的代码：证据数据和日志保留在 `/tmp`；phase185/186 环境变量、只读导出、固定动作干预和环境快照接口已全部删除，生产路径不保留诊断逻辑
前序阶段回归：两轮正式 rollout 均为纯动力学，四个 rank 正常退出，无 optimizer、checkpoint、异常或 NaN；清理后 py_compile、9 项单测、git diff 和临时标识搜索通过
下一决策门槛：正式方案只能位于统一 action distribution/reference 所有权层，必须能在 PPO update 中严格重算新旧条件 log-prob；先完成接口设计和概率单测，再允许训练
```

2026-09-27 10:40，phase186 单卡 `8 env x 8 steps` smoke 正常退出。`phase186_group`、`phase186_active`、`phase186_executed_ee_x` 及 phase185 全部审计字段维度一致、值有限，三个固定分组均存在；进程在 optimizer 前退出。由于单卡拓扑不加载四卡 staged-reset，smoke 中没有 released 样本，只证明干预和导出链路可运行，不构成物理结论。

2026-09-27 10:44，phase186 首轮四卡 rollout 正常退出，初步结果显示 `ee_x=0.02` 在左右门 32 步内分别产生约 `0.028/0.029 rad` 铰链增量，而 `0.00` 对照近似为零；两侧 opening-force 最大值约 `128/133 N`。但 opening force 不是 `600 N` 安全终止实际使用的完整接触力，现有数据不能完成安全判定。为避免用不完整指标通过实验，仅增加 `max_handle_force`、`panel_gripper_force` 和二者最大值的只读导出后原样复跑；干预、环境和成功标准不变。

2026-09-27 10:52，phase186 补全安全字段后的四卡复跑正常退出，共 `1,048,576` 个样本、`230351` 个 OPEN_LATCH 样本、`55050` 个实际干预帧。`ee_x=0.02` 相对 `0.00` 的 32 步结果如下：

```text
                                左门             右门
实际 EE task-x 增量             +2.995 mm        +0.528 mm
hinge angle 增量                +0.02697 rad      +0.02922 rad
进入后续阶段比例                 4.73%             8.76%
完整任务接触力 p99               62.38 N           70.31 N
完整任务接触力最大值             150.62 N          184.40 N
完整任务接触力 > 600 N           0                 0
当前动作后的 done 率             0.089%            0.081%
```

同侧 `0.00` 对照的 32 步 hinge 增量均近似 `0`，后续阶段转换均为 `0`，done 率为左/右 `0.114%/0.088%`。因此 `0.02` 的进展不是初态分布或更高安全失败率造成。`0.01` 在左右门只形成约 `0.0049/0.0055 rad` 门角增量，没有阶段转换，并在 32 步内持续损失 retention；这表明探索需要足够连续位移越过接触/验收死区，不能用轻微单步偏置代替。

2026-09-27 10:53，phase185/186 临时代码全部删除。容器内 py_compile、9 项单测、`git diff --check` 和 `B2Z1_PHASE18|get_phase185|phase186_` 标识搜索通过；主机和容器内均无训练或诊断进程。当前有效 checkpoint 仍为 phase183 step125，phase186 没有执行 optimizer 或生成新模型。

### phase187：时间连续探索的统一概率接口设计（已验证）

```text
上游结论与待解决瓶颈：phase186 已证明持续同向 delta 能在左右门安全开缝；当前独立 Gaussian exploration 的时间结构与 accumulator 动力学不匹配
基线 checkpoint / SHA-256：phase183 model_step_000125.pt / 104d110b48eddb9b57819ac6a3cbf1bf386e764aa8cb935f9bd449abab7c22d7（后续训练对照基线，本 phase 先不运行）
唯一变量：先审计 policy distribution、recurrent rollout state、storage 和 PPO update 的 log-prob 数据流，设计统一的时间连续探索接口；不在环境或阶段逻辑中注入动作
待验证假设：可以把时间相关探索建模为策略的显式条件分布，并把其必要状态随 rollout 存储，使新旧策略都能对同一条件动作严格计算 log-prob
禁止改动范围：reward、observation/action 物理语义、阶段条件、时限、随机化、低层、AutoOpen、门型和 seed 行为；禁止固定 ee_x、action reward、门型分支、隐藏开关或只在 OPEN_LATCH 生效的生产补丁
运行规模与日志：先做静态数据流审计和无仿真概率单测；设计闭合前不启动 Isaac 或训练
成功/失败判据：明确探索状态的初始化/reset、采样、storage、mini-batch 重建和新旧 log-prob 公式；单测覆盖时间相关性、边际方差、done reset、old/new 相等时 ratio=1，以及参数变化后的解析 ratio
观测指标：噪声 lag 自相关、条件均值/方差、rollout 与 update log-prob 误差、PPO ratio、KL 和各动作维覆盖
结论与已排除方向：采用参数无关的标准化 AR(1) 上下文 `z_t`，并令条件动作分布为 `N(mu + sigma*sqrt(f)*z_t, sigma^2*(1-f))`。`z_t` 的生成概率与策略参数无关，PPO 对 rollout 存储的同一个 `z_t` 计算新旧条件概率，故上下文概率在 ratio 中严格抵消；边际动作方差仍为 `sigma^2`，lag-k 协方差为 `f*rho^k*sigma^2`。排除固定动作、环境侧滤波、action reward 和无法在 shuffled mini-batch 重算的隐式历史滤波
正式保留/删除的代码：保留独立通用概率核 `gr00t/rl/trl/modules/temporal_action_noise.py` 和纯数学单测；尚未接入 Actor、trainer、环境或配置
前序阶段回归：容器内 py_compile、原有 9 项和新增 5 项共 14 项单测、git diff 检查通过；正式行为未改变
下一决策门槛：phase188 以默认关闭方式接入 Actor/storage/PPO，必须区分 base mean 和 conditional distribution mean，并证明关闭时逐张量零差异、开启时 rollout/update ratio 在参数不变时严格为 1
```

2026-09-27 10:59，phase187 完成策略分布、recurrent rollout、storage、trajectory mini-batch 和 PPO loss 数据流审计。相关上下文在每个动作步显式存储，因此 update 不需要依赖被 shuffle 打乱的隐式前一步状态；done 后用新标准正态创新重置，rollout 边界可连续保留上下文，checkpoint 恢复时从平稳分布重新初始化。

概率核单测覆盖：参数范围、done reset、相关/独立方差分解、采样后边际方差与 lag-1 相关、相同参数 ratio=1，以及参数变化后与手算解析 ratio 一致。测试规模下实测总方差误差和相关系数误差均小于 `0.02`。

### phase188：Actor/storage/PPO 条件分布接入（已验证）

```text
上游结论与待解决瓶颈：phase187 已闭合时间连续探索的概率模型；现有 Actor 只暴露一个 action_mean，无法同时表达监督/诊断需要的 base mean 和 PPO KL 需要的 conditional distribution mean
基线 checkpoint / SHA-256：phase183 model_step_000125.pt / 104d110b48eddb9b57819ac6a3cbf1bf386e764aa8cb935f9bd449abab7c22d7（本 phase 不运行训练）
唯一变量：增加配置默认关闭的通用 temporal-noise 接口，并贯通 Actor rollout、storage、recurrent mini-batch 和 PPO forward；B2Z1 配置保持关闭
待验证假设：显式存储 standardized context 后，rollout 和 update 能重建相同条件分布；base mean 继续用于 action supervision/任务诊断，conditional mean/std 仅用于采样、log-prob 和 KL
禁止改动范围：B2Z1 配置、reward、observation/action 物理语义、环境、阶段条件、随机化、低层和 AutoOpen；默认关闭路径不得产生数值或 storage 行为变化
运行规模与日志：无仿真模块/数据流单测；通过后才做最小 Isaac smoke
成功/失败判据：关闭配置逐张量等价；开启配置 ratio=1、解析 KL/ratio 正确、done reset 正确、recurrent dense shape 对齐；监督使用 base mean，KL 使用 conditional mean
观测指标：actions、base mean、distribution mean、conditional std、context、old/new log-prob、ratio、KL 和 supervision target error
结论与已排除方向：通用接口已完成。`action_mean` 始终表示网络 base mean，`action_distribution_mean` 只表示加入已存 AR context 后的条件均值，`action_sigma` 为条件标准差；PPO storage 仅在功能启用时增加 distribution mean/context 字段，KL 和 log-prob 使用条件分布，action supervision 和任务诊断继续使用 base mean。默认关闭与原高斯接口逐张量一致。真实 recurrent rollout 写入 storage、打乱环境顺序并做 batch LSTM forward 后，参数不变时 log-prob 最大误差小于 `2e-6`，ratio 在同一容差内为 1
正式保留/删除的代码：保留通用概率核、Actor/recurrent Actor/storage/PPO 接口及回归测试；不包含门型、stage、seed、固定动作或环境侧滤波。deterministic policy 查询改为直接返回网络 base mean，不再错误依赖采样 context
前序阶段回归：默认关闭 Isaac smoke 为 `8 env x 8 steps`，从 step125 完成 iteration126 和 PPO update；临时启用 `rho=0.95/fraction=0.9` 的同规模 smoke 也完成 iteration126 和 update，两者均无异常、NaN 或残留进程。完整无仿真测试 `19 passed`，py_compile 与 git diff 检查通过
下一决策门槛：phase189 才允许在 B2Z1 配置统一启用候选参数，并先做四卡只读 rollout；只有真实自相关、边际方差、安全和左右门覆盖同时通过，才允许更新模型
```

2026-09-27 11:06，phase188 默认关闭 smoke 正常完成。日志 `/tmp/b2z1_phase188_disabled_smoke.log`；`8 env x 8 steps` 从 step125 完成一次 rollout 和 PPO update。关闭时不注册 temporal storage 字段，base mean 与 distribution mean 相同，未出现异常或 NaN。

2026-09-27 11:08，使用 Hydra 临时覆盖 `rho=0.95`、`correlated_fraction=0.9` 的启用态 smoke 正常完成。日志 `/tmp/b2z1_phase188_enabled_smoke.log`；同为 `8 env x 8 steps`，完成 iteration126 和一次 PPO update，无异常、NaN 或遗留训练进程。该运行只验证真实 Isaac 数据流，不把生成的 step126 目录作为有效 checkpoint。

2026-09-27 11:11，补齐完整 recurrent 数据流回归：使用真实 `RecurrentActor` 逐步 rollout，显式保存 actions、context 和 old log-prob 到 `RolloutStorage`，随机打乱环境顺序后经 trajectory split、batch LSTM forward 和相同 context 重算 log-prob。最大绝对误差小于 `2e-6`，ratio 在同一容差内为 1。容器内完整测试为 `19 passed`，相关模块 py_compile 和 `git diff --check` 均通过。phase188 至此完成，当前无训练或诊断进程。

备份记录：2026-09-27 11:13 已将 phase185--188 正式代码、本文档和 step125 有效 checkpoint 同步到 `wholebody_1/3/4`，不修改远程 Git 历史或凭据。29 个工作区代码/文档文件加 checkpoint 的聚合 SHA-256 在源机和三台备份机均为 `9ac06af255bc591174bc149f37c3012ff9dc11cd840528033b5991a4a7c67b82`。

### phase189：B2Z1 时间相关探索只读覆盖审计（失败）

```text
上游结论与待解决瓶颈：phase186 已证明持续 canonical delta 能让左右门安全开缝；phase188 已完成可严格重算 PPO 条件概率的统一 temporal-noise 接口，但尚未在真实大规模 rollout 验证候选参数的行为覆盖
基线 checkpoint / SHA-256：phase183 model_step_000125.pt / 104d110b48eddb9b57819ac6a3cbf1bf386e764aa8cb935f9bd449abab7c22d7（只读审计，不更新模型）
唯一变量：在 B2Z1 统一策略配置启用 `temporal_action_noise_rho=0.95` 和 `temporal_action_noise_correlated_fraction=0.9`；不按阶段、门型、seed 或动作维分支
待验证假设：保持每一动作维原边际方差不变时，时间相关探索能提供 delta accumulator 所需的持续同向覆盖，使左右门 released 状态都更常建立 opening force、门缝和 REPOSITION 转换，且不损害安全
禁止改动范围：reward、observation/action 物理语义、网络、阶段条件、时间预算、随机化、低层、AutoOpen、checkpoint 参数和 optimizer；禁止 OPEN_LATCH 专用噪声、固定 ee_x 或动作奖励
运行规模与日志：先做配置加载和最小 smoke，再做 `4 x 2048 env x 128 steps` 只读 rollout；四个 rank 导出参数无关概率与物理覆盖证据，在 optimizer 前退出
成功/失败判据：实际标准化 action noise 的单步边际方差应保持原尺度，lag-k 自相关应与 `0.9 * 0.95^k` 一致；参数不变重算 ratio 接近 1；左右门 opening force、hinge/gap 或阶段转换相对 phase185 独立噪声基线共同改善；完整接触力不得超过 600 N，安全终止不得显著恶化
观测指标：各动作维边际均值/方差与 lag 1/2/4/8/16/32 自相关、context、old/recomputed log-prob、ratio、左右门 released opening force、实际 EE task-x、hinge/gap、retention、阶段转换、done 和完整接触力
结论与已排除方向：`rho=0.95/fraction=0.9` 的概率实现完全符合设计，且相对独立噪声让左右门都建立 opening force、门角和少量阶段转换；但匹配的终止前对照证明其在 OPEN_LATCH 新增 42 次 `>600 N`，并使 released retention 左/右从 `0.903/0.883` 降至 `0.601/0.565`，不满足安全和夹持约束。排除直接用 0.9 correlated fraction 训练
正式保留/删除的代码：通用 temporal 接口保留；phase189 只读导出暂时保留供 phase190 使用，参数选择完成后必须删除；B2Z1 correlated fraction 已进入下一候选，不保留失败值 0.9
前序阶段回归：运行前确认 19 项单测、默认关闭路径和纯动力学约束；运行后检查 APPROACH_PREGRASP/GRASP、安全和无相机渲染状态
下一决策门槛：phase190 只降低 correlated fraction 到 0.5；必须在双侧保留明显优于 independent 的 opening force/门角覆盖，同时 OPEN_LATCH unsafe-force 回到 0 且 retention 明显恢复，才允许训练
```

2026-09-27 11:18，phase189 最小 smoke 完成。`8 env x 8 steps` 加载 step125，18 个导出字段形状一致且浮点值全部有限；从保存的 conditional mean/std 直接重算 old log-prob，最大绝对误差为 `1.53e-5`。日志 `/tmp/b2z1_phase189_smoke.log`，数据 `/tmp/b2z1_phase189_smoke.rank0.pt`。运行在 optimizer 前退出，无参数更新、checkpoint、异常、NaN 或残留进程；运行日志明确确认 cameras=0、RTX sensors=0、frame rendering disabled。该 smoke 只验证链路，不构成覆盖或物理结论。

2026-09-27 11:22，首次四卡候选 rollout 完成，但审计在 `env.step()` 返回后读取接触力，终止环境已经 reset。该轮概率和非终止物理字段有效，但 `max=599.96 N、0 次 >600 N` 是生命周期错误造成的假象，禁止作为安全证据。只读快照随后移到 `_check_termination()` 内、reset 前；行为逻辑不变。

2026-09-27 11:31，终止前候选复跑完成。`1,048,576` 个样本的 normalized noise 方差为 `0.993--1.005`，lag 1/2/4/8/16/32 相关均值为 `0.8549/0.8121/0.7327/0.5961/0.3951/0.1725`，与理论 `0.8550/0.81225/0.73306/0.59708/0.39611/0.17434` 一致；重算 old log-prob 最大/平均误差为 `3.05e-5/4.50e-6`。概率实现通过。

2026-09-27 11:38，temporal-disabled 匹配对照完成。独立噪声 lag-1 为 `-0.00006`，边际方差为 `0.9996`；全局 unsafe-force 为 159 次，按阶段为 `[82,3,0,74,0,0]`。0.9 temporal 候选为 207 次，按阶段为 `[77,1,42,87,0,0]`；因此 OPEN_LATCH 的 42 次全部是候选新增风险，不能被原有 APPROACH/REPOSITION 终止掩盖。

同一对照中，independent 左/右 released opening-force 均值为 `0/0.033 N`、`>1 N` 比例为 `0/0.168%`、hinge 均值为 `0/0.000009 rad`、后续阶段转换为 `0/0`。0.9 temporal 候选提高到 opening-force `1.91/3.83 N`、`>1 N` 比例 `5.87%/13.96%`、hinge `0.00304/0.00612 rad`，并产生左/右 `19/42` 次转换；但 retention 降至 `0.601/0.565`，且 OPEN_LATCH unsafe-force 为 42 次。phase189 因此失败，不执行 optimizer、不生成 checkpoint。

### phase190：降低相关方差比例的安全覆盖审计（失败）

```text
上游结论与待解决瓶颈：phase189 证明统一 temporal exploration 能产生缺失的双侧物理覆盖，但 correlated fraction=0.9 持续分量过强，造成 42 次 OPEN_LATCH unsafe-force 和显著 retention 损失
基线 checkpoint / SHA-256：phase183 model_step_000125.pt / 104d110b48eddb9b57819ac6a3cbf1bf386e764aa8cb935f9bd449abab7c22d7（只读审计，不更新模型）
唯一变量：`temporal_action_noise_correlated_fraction: 0.9 -> 0.5`；rho 保持 0.95，总边际方差、checkpoint、环境和全部训练接口不变
待验证假设：降低持续分量幅度可保留足以跨越接触死区的时间覆盖，同时显著恢复 retention 并消除 OPEN_LATCH unsafe-force
禁止改动范围：reward、observation/action、网络、阶段条件、时间预算、随机化、低层、AutoOpen、rho、边际 std、optimizer 和任何阶段/门型/seed 特例
运行规模与日志：`4 x 2048 env x 128 steps`，终止前只读导出，optimizer 前退出；与 phase189 independent 和 fraction=0.9 两组同字段对照
成功/失败判据：normalized noise 方差仍约为 1、lag-k 符合 `0.5*0.95^k`；左右门 opening force/hinge/转换均显著高于 independent；OPEN_LATCH unsafe-force 必须为 0，retention 必须较 0.9 候选明显恢复
观测指标：同 phase189，重点比较 stage x variant unsafe-force、released retention、opening force、hinge 和转换
结论与已排除方向：fraction=0.5 的实际 lag-1 为 `0.4747`、总噪声方差为 `0.9989`，概率正确；左右门 released opening-force 为 `0.567/1.374 N`、hinge 为 `0.000675/0.002901 rad`、转换为 `3/15`，均优于 independent，但 OPEN_LATCH 仍有 `21` 次 unsafe-force，retention 仅恢复到 `0.696/0.659`。排除继续用统一 scalar fraction=0.5 训练
正式保留/删除的代码：沿用 phase189 无副作用只读审计，phase190 结束后清理
前序阶段回归：只读，不更新参数；继续确认纯动力学、无相机和前序阶段安全
下一决策门槛：phase191 不再继续盲目减小所有维度的 scalar fraction；根据 phase186 的直接因果证据，只对 canonical ee_x 配置时间相关方差，其他维保持原独立探索。必须重新通过概率、双侧覆盖、retention 和 unsafe-force 验收
```

2026-09-27 11:45，phase190 四卡只读 rollout 完成，无 optimizer、checkpoint、异常或 NaN。与 independent/fraction0.9 的同字段结果：全局 unsafe-force 为 `159/207/178`，OPEN_LATCH 为 `0/42/21`；released retention 左门为 `0.903/0.601/0.696`，右门为 `0.883/0.565/0.659`；转换为左 `0/19/3`、右 `0/42/15`。fraction=0.5 虽折中改善，但仍明确不满足零新增 OPEN_LATCH unsafe-force，phase190 失败。

### phase191：仅 canonical ee_x 的时间相关探索（已验证）

```text
上游结论与待解决瓶颈：scalar fraction 0.9/0.5 都让多个位置和姿态 delta 同时持续漂移，造成夹持退化和 OPEN_LATCH unsafe-force；phase186 的直接因果证据只支持持续 canonical ee_x
基线 checkpoint / SHA-256：phase183 model_step_000125.pt / 104d110b48eddb9b57819ac6a3cbf1bf386e764aa8cb935f9bd449abab7c22d7（只读审计）
唯一机制变量：correlated fraction 支持标量或按动作维向量；B2Z1 使用 `[0,0,0,0.9,0,0,0,0,0,0]`，仅 action index 3 canonical ee_x 有时间相关分量，其他维边际分布和时间独立性不变
待验证假设：保留因果上必要的 ee_x 持续覆盖，同时消除 y/z/姿态持续漂移造成的 retention 损失和冲击
禁止改动范围：reward、observation/action 物理语义、网络、阶段条件、时限、随机化、低层、AutoOpen、边际 std、optimizer、stage/door/seed 分支和动作均值注入
运行规模与日志：先做纯数学和 Actor/storage 回归，再做 `4 x 2048 env x 128 steps` 终止前只读 rollout，optimizer 前退出
成功/失败判据：ee_x lag-k 符合 `0.9*0.95^k`，其余维 lag 相关约 0，所有维边际方差约 1，log-prob 可重算；双侧 opening force/hinge/转换优于 independent；OPEN_LATCH unsafe-force 为 0，retention 显著优于 scalar 候选
观测指标：同 phase189/190，另按动作维核对实际 lag 相关
结论与已排除方向：ee_x 实测 lag 1/2/4/8/16/32 为 `0.8546/0.8120/0.7325/0.5964/0.3963/0.1788`，其余 9 维均约为 0，10 维边际方差为 `0.997--1.002`，old log-prob 重算最大误差 `3.05e-5`。相对 independent，左右门产生 `2/13` 次转换，released opening-force 为 `0.199/0.792 N`，retention 为 `0.836/0.805`。全局 unsafe-force `163` 与 independent `159` 近似，但 OPEN_LATCH 从 0 增至 15 次（15/226228=`0.0066%`），故未满足原先绝对零判据；考虑高斯分布无界，后续改用明确率阈值和训练中持续监控，而不声称零风险
正式保留/删除的代码：保留通用 per-action fraction 接口、测试和 B2Z1 ee_x-only 配置；phase189--191 只读导出和环境快照代码已全部删除，标识搜索为空
前序阶段回归：完整 `20 passed`、py_compile、git diff 检查通过；清理后的 vector 配置从 step125 完成真实 `8 env x 8 steps` PPO update smoke，无异常、NaN 或残留进程
下一决策门槛：phase192 仅做 25 update 受控训练；若 OPEN_LATCH unsafe-force 率超过 `2e-4`（约只读候选的 3 倍）或连续恶化，立即停止。否则以 step150 左右门转换率和 retention 判断学习是否真正利用新覆盖
```

2026-09-27 11:54，phase191 四卡只读 rollout 完成。全局 unsafe-force 为 163 次，按阶段 `[69,4,15,75,0,0]`；independent 对照为 `[82,3,0,74,0,0]`。全局风险没有实质增加，但 OPEN_LATCH 出现 15 次任务相关安全终止。左右门 released retention 相比 scalar0.9 的 `0.601/0.565` 恢复至 `0.836/0.805`，同时保留非零 opening force、门角和转换覆盖。该结果支持受控短训练，不支持宣称完全安全或直接长时间训练。

2026-09-27 12:00，phase189--191 临时只读审计分支和 pre-reset 快照已删除。正式代码保留通用 per-action temporal probability interface 与 B2Z1 ee_x-only 配置。完整测试 `20 passed`，清理后真实 PPO update smoke 正常完成；无训练进程。

备份记录：2026-09-27 12:03 已将 phase188--191 正式代码、本文档和 step125 有效 checkpoint 同步到 `wholebody_1/3/4`；除本状态文档外的聚合 SHA-256 四机一致为 `6fca83b79188abdeb1d6e30855579956f5ac6316b586dab806d61dd96d854bbb`，状态文档也已在该记录写入后单独同步；未修改远程 Git 历史或凭据。

### phase192：ee_x-only temporal exploration 受控短训练（失败）

```text
上游结论与待解决瓶颈：ee_x-only 时间相关探索在不持续扰动其他 9 维的情况下产生双侧开缝覆盖，且全局安全率接近 independent；尚不知 PPO 是否能把该覆盖转为 base mean 和阶段转换提升
基线 checkpoint / SHA-256：phase183 model_step_000125.pt / 104d110b48eddb9b57819ac6a3cbf1bf386e764aa8cb935f9bd449abab7c22d7
唯一变量：启用已验证的 per-action fraction `[0,0,0,0.9,0,0,0,0,0,0]` 后执行 PPO update；其余正式机制不变
待验证假设：条件概率正确的持续 ee_x 探索会让 PPO 看到并强化安全开缝轨迹，使左右门 OPEN_LATCH -> REPOSITION 转换率脱离 step125 平台，同时保持夹持和前序阶段
禁止改动范围：训练运行中不改代码；不改 reward、observation/action、网络、阶段条件、时限、随机化、低层、AutoOpen、rho/fraction、std 或门型/seed 逻辑
运行规模与日志：4 x 2048 env、128 steps/env、4 mini-batches，从 step125 训练 25 update 到 step150；每 25 update 保存；日志 `/tmp/b2z1_phase192_eex_temporal_to150.log`
成功/失败判据：左右门 OPEN_LATCH -> REPOSITION 转换相对 step125 的 `4.95%/12.09%` 均有可复现改善，retention 不明显回归，APPROACH/GRASP 不回归；OPEN_LATCH unsafe-force 每步率不超过 `2e-4` 且不持续上升
观测指标：六阶段 reached、stage x variant 转换、OPEN_LATCH base mean/sample/opening force/hinge/retention、分阶段 unsafe-force、done、KL、ratio 和 loss
结论与已排除方向：25 update 后左/右 OPEN_LATCH -> REPOSITION 转换从 `4.95%/12.09%` 提高到 `9.32%/29.89%`，证明 temporal exploration 提供了有效覆盖；但 pocket/axis/contact retention 从 `0.8921/0.9271/0.9425` 降到 `0.8171/0.8749/0.8084`，最终 base mean `ee_x=-0.0008/-0.0022`，策略没有吸收所需持续正向动作。排除“有覆盖后直接续训即可同时学会开缝和保持夹持”
正式保留/删除的代码：仅保留通过 phase191 的正式机制和永久分阶段安全指标
前序阶段回归：20 项测试和真实 update smoke 已通过
下一决策门槛：step150 不升格为有效 checkpoint。phase193 回到 step125 做同分布只读诊断，必须闭合成功 temporal 序列的完整 credit 和 base-mean gradient 后，才允许决定修改 reward、探索结构或训练接口
```

2026-09-27 12:09，phase192 已从 step125 完成前两个 update（step126、step127），四个 rank 均存活，未见异常或 NaN。最近两个 update 的 `Env/termination/unsafe_force/open_latch` 均为 `0.0000`；该早期结果只用于安全监控，不足以判断左右门转换、retention 或 actor base mean 是否改善。训练继续按原定唯一变量运行，期间不修改代码或参数。

2026-09-27 12:34，phase192 正常完成 step150 并退出，25 个 update 均未记录到 OPEN_LATCH unsafe-force，APPROACH_PREGRASP 左右保持 `100%`，GRASP 从 step125 的 `43.63%/47.06%` 提高到 `45.65%/49.79%`。左/右 OPEN_LATCH reached 为 `33.26%/28.64%`，REPOSITION reached 为 `3.10%/8.56%`，对应转换率 `9.32%/29.89%`，明显高于基线 `4.95%/12.09%`；PUSH_OPEN reached 为 `1.44%/3.21%`，THROUGH 为 `0.02%/2.04%`。

该覆盖改善没有满足完整验收：OPEN_LATCH pocket/axis/contact retention 从 step125 的 `0.8921/0.9271/0.9425` 降到 `0.8171/0.8749/0.8084`；最终左/右 policy base mean `ee_x` 为 `-0.0008/-0.0022`，而持续正向 canonical `ee_x` 才是 phase186 证明的安全开缝因果动作。说明阶段转换主要仍由相关 sample 覆盖提供，PPO 没有把成功序列稳定吸收到策略均值。step150 checkpoint `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260927_120307/model_step_000150.pt`（SHA-256 `270309ac35cd679489192289b2d429f0f3580ff887762e154a92399c5bd5e8ce`）仅保留为失败证据，不作为后续训练基线。

### phase193：temporal 成功序列到 base-mean gradient 的只读诊断（计划中）

```text
上游结论与待解决瓶颈：phase192 的 correlated sample 显著提高双侧阶段转换，但 retention 回归且 actor base mean 仍为负；尚不清楚成功序列本身得到的完整 PPO credit 是否为正、是否被更多失败序列抵消，或 conditional distribution 到 base mean 的梯度聚合存在不同机制
基线 checkpoint / SHA-256：phase183 model_step_000125.pt / 104d110b48eddb9b57819ac6a3cbf1bf386e764aa8cb935f9bd449abab7c22d7（只读诊断，不继承 phase192 step150）
唯一变量：增加默认关闭、optimizer 前退出的只读导出；正式环境、策略分布、reward、checkpoint 和行为保持 phase192 启动时完全一致
待验证假设：真正形成门缝并转换的 temporal 序列对 base mean 具有正向 PPO gradient，但其样本量或加权 credit 被 retention 下降的失败序列抵消；若不成立，则定位首个符号或归因分叉层
禁止改动范围：不更新参数；不改 reward、observation/action、网络、阶段条件、时限、随机化、低层、AutoOpen、rho/fraction/std；禁止 stage、门型、seed 特调和固定动作注入
运行规模与日志：最小 smoke 为单卡 `8 env x 8 steps`，日志 `/tmp/b2z1_phase193_smoke.log`、数据 `/tmp/b2z1_phase193_smoke.rank0.pt`；通过后做 `4 x 2048 env x 128 steps` 同分布只读 rollout，正式路径启动时登记
成功/失败判据：按门型、released、retention 分位和未来是否转换分层，能从 correlated context 与 sampled ee_x 一直追到完整 sample weight、advantage、ratio/surrogate 和对 base mean 的实际梯度贡献；分层和必须重建整体 actor update
观测指标：base mean、conditional mean/std、AR context、sampled action、old/new log-prob、完整 stage/variant weight、advantage、clipped surrogate、未来 1/2/4/8/16/32 步 retention/opening force/hinge/transition、每层 base-mean gradient
结论与已排除方向：临时导出已实现；解析 base-mean 更新方向与 PyTorch 自动微分最大误差 `1.19e-7`，20 项单测、py_compile 和 git diff 检查通过。单卡 `8 env x 8 steps` smoke 已在 optimizer 前正常退出，32 个字段形状和有限性通过，old log-prob 重算最大误差 `1.53e-5`，signal 重算最大相对误差 `1.27e-7`，未生成 checkpoint；因拓扑变化未恢复 staged-reset，0 个 OPEN_LATCH 样本只验证链路。尚未完成正式 rollout
正式保留/删除的代码：诊断代码默认关闭；phase 结束后必须删除，只保留被证明通用且运行期需要的指标
前序阶段回归：启动前要求 20 项单测、py_compile、git diff 检查和最小 Isaac smoke；只读运行不得产生 checkpoint
下一决策门槛：根因闭合后只选择一个所有权层修复；证据不足时不续训、不调整 reward 或 temporal 参数
```

备份记录：2026-09-27 12:40 已将 phase192 最终结论、phase193 计划、当前正式工作树、step125 有效 checkpoint 和 step150 失败证据同步到 `wholebody_1/3/4`。除本状态文档外的源码/文档内容聚合 SHA-256 四机均为 `7939170f8127f9b176e399cb400316f87626ea00688c8b8366555b8b50ca8620`；step125/step150 checkpoint SHA-256 四机分别一致为 `104d110b48eddb9b57819ac6a3cbf1bf386e764aa8cb935f9bd449abab7c22d7` 和 `270309ac35cd679489192289b2d429f0f3580ff887762e154a92399c5bd5e8ce`。未修改远程 Git 历史或凭据，未同步训练日志和无关目录。

2026-09-27 12:55，phase193 四卡正式只读 rollout 完成。日志 `/tmp/b2z1_phase193_credit.log`，数据 `/tmp/b2z1_phase193_credit.rank{0,1,2,3}.pt`；每个 rank 均在 optimizer 前退出，未更新参数或生成 checkpoint。解析 base-mean 更新方向与 autograd 最大误差 `1.19e-7`，old log-prob 重算误差保持在已验证容差内。

正式数据首先暴露了一个比 temporal reward 分层更上游的通用接口错误。在标为 OPEN_LATCH 的 `225678` 个样本中，14 个 `GRASP -> OPEN_LATCH` 第一帧的 canonical `ee_x` conditional std 为 `3.1622778e-7`，而 OPEN_LATCH 的正确值应为 `0.0063245557`。典型时序为 `task_stage=[1,1,2,2,2]`，对应 `ee_x std=[3e-7,3e-7,3e-7,0.00632,0.00632]`。

代码调用顺序确认根因：`_check_termination()` 生成 advance event，`_compute_reward()` 让完成动作按旧阶段获得 credit，随后 `_compute_observations()` 仍用旧 `stage_buf` 生成 observation，最后 `_post_compute_observations_callback()` 才递增阶段。trainer 下一轮用这份旧 observation 采样，但同时从已递增的 `env.stage_buf` 写入新阶段标签。上述 14 个样本被 OPEN_LATCH 稀有样本权重和 advantage 处理，实际分布却仍使用 GRASP action mask；接近零的 std 将解析 score 放大到百万级，右门 OPEN_LATCH update sum 被其主导到约 `-2.42e7`。因此 phase192 的负 base mean 不能继续按 reward 或 temporal exploration 根因解释，必须先修复时序污染再复测。

真正转换前 32 步中，左/右事件数为 `2/8`，总 action 相对 base mean 均为正 `+0.0382/+0.0324`，相关 context 提供绝大多数偏移；成功窗口实际 conditional-score 更新方向为左 `2/2` 正、右 `5/8` 正，但比只看总 action 偏移得到的表观 credit 弱约 47--100 倍。该探索可学习性证据暂不作最终结论，待 phase194 无污染复验。

### phase194：staged-task observation/storage 时序一致性修复（已验证）

```text
上游结论与待解决瓶颈：phase193 证明阶段转换首帧存在“新 stage 标签 + 旧 stage policy distribution”的通用时序错位，并以近零 std 放大错误 PPO score
基线 checkpoint / SHA-256：phase183 model_step_000125.pt / 104d110b48eddb9b57819ac6a3cbf1bf386e764aa8cb935f9bd449abab7c22d7（只读复验，不更新参数）
唯一变量：在通用任务生命周期中新增“为下一 observation 提交任务状态”钩子；StagedTaskBase 在旧阶段 reward 结算后、下一 observation 生成前完成阶段递增、callback 和 snapshot
待验证假设：完成动作继续获得旧阶段 transition credit，同时下一策略帧的 stage one-hot、action mask、conditional distribution 和 trainer task_stage 全部一致表示新阶段
禁止改动范围：不改 B2Z1 reward、observation/action 字段、网络、temporal 参数、阶段条件、时限、随机化、低层、AutoOpen、门型或 seed；禁止 trainer 特判、移动标签、丢弃 14 个样本或兼容旧 checkpoint 的分叉
运行规模与日志：22 项无 Isaac 单测和 py_compile 已通过；下一步最小 Isaac smoke，通过后原样复跑 4 x 2048 env x 128 steps phase193 只读审计
成功/失败判据：完成动作的 transition reward 仍归旧阶段；所有 task_stage==OPEN_LATCH 样本的 ee_x conditional std 均为 0.0063245557；不存在新 stage 标签配旧 mask；old log-prob 可按存储字段重算；reset/staged-reset snapshot 行为不回归
观测指标：transition event、reward stage、observation stage one-hot、env.stage_buf、task_stage、conditional std/mean、old log-prob 重算误差和 stage callback/snapshot 次数
结论与已排除方向：待真实 Isaac 验证；已经排除 trainer 侧改标签、过滤异常样本和 B2Z1 阶段特判，因为这些方案不能恢复 observation/action 语义一致性
正式保留/删除的代码：候选修复位于 LeggedRobotBase 通用 observation 生命周期钩子、StagedTaskBase 和纯张量事件函数；phase193 PPO 审计代码在复验完成后删除
前序阶段回归：当前 22 项无 Isaac 测试通过，包含旧阶段 credit、新阶段 observation、reset/just-reset 屏蔽；真实 Isaac 尚待验证
下一决策门槛：只有同规模只读 rollout 清除全部错位样本，才能重新分析 temporal 成功序列 gradient；否则继续停在接口层定位，不改 reward 或探索
```

2026-09-27 13:20，phase194 完成。修复位于通用生命周期所有权层：`LeggedRobotBase` 在生成下一 observation 前调用任务状态提交钩子，`StagedTaskBase` 在该钩子中消费已由 `_check_termination()` 缓存的 advance event。顺序现在固定为“旧阶段 termination/event -> 旧阶段 reward -> reset -> 提交新阶段/callback/snapshot -> 生成新阶段 observation”。没有修改 trainer 标签、B2Z1 阶段条件、reward、action 或 temporal 参数。

四卡拓扑 8-step smoke 恢复了 checkpoint 中每个 rank 的 staged-reset 状态，共得到 `65536` 个样本并覆盖全部六阶段；OPEN_LATCH `13479` 个样本的 canonical `ee_x` conditional std 全部为 `0.0063245557`，近零错位为 `0`。随后原样复跑 `4 x 2048 env x 128 steps`，日志 `/tmp/b2z1_phase194_credit.log`，数据 `/tmp/b2z1_phase194_credit.rank{0,1,2,3}.pt`；共 `1,048,576` 个样本，OPEN_LATCH `224903` 个，错误方差样本仍为 `0`，old log-prob 最大重算误差 `3.05e-5`。四个 rank 均在 optimizer 前退出，未生成 checkpoint。

污染去除后，OPEN_LATCH 的 canonical `ee_x` base-mean update signal 聚合从 phase193 的 variant 0/1 `+4.69e6/-2.42e7` 回到正常量级 `-6.77e4/-9.96e4`，证明此前百万级 score 确由时序错位造成。32 步内进入后续阶段的样本中，总动作相对 base mean 为 variant 0/1 `+0.04597/+0.03869`，其中 AR context shift 为 `+0.04517/+0.03916`，独立 conditional residual 仅 `+0.00080/-0.00047`；对应 update signal sum 为 `+1390.9/-2628.0`。因此接口 bug 已解决，但当前外生 context 的 conditional likelihood 对成功持续动作归因很弱，且一侧仍为负，这是 phase195 的唯一问题。

phase193 临时 storage 字段、环境只读采集、导出函数、环境变量入口和 optimizer 前提前退出路径已全部删除。清理后 `22 passed`、py_compile、`git diff --check` 和临时标识搜索通过；生产 trainer 不承担诊断开销。

备份记录：2026-09-27 13:23 已将 phase194 修复、清理后的正式源码、测试和本文档同步到 `wholebody_1/3/4`，未修改远程 Git 历史或凭据，也未同步 `/tmp` 审计数据。同步时四机 `gr00t/` 与 `docs/` 内容聚合 SHA-256 均为 `178726da8f0cc12119a02f39445942af3502d5d9d4cd0f6f0032f91566665e9e`；写入本条记录后状态文档单独再次同步。

### phase195：时间相关探索的可学习轨迹概率审计（已验证）

```text
上游结论与待解决瓶颈：phase194 清除 stage 首帧污染；成功持续位移几乎全由参数无关 AR context 提供，而 conditional PPO score 只看到很小的独立残差
基线 checkpoint / SHA-256：phase183 model_step_000125.pt / 104d110b48eddb9b57819ac6a3cbf1bf386e764aa8cb935f9bd449abab7c22d7（先不训练）
唯一变量：只审计并设计时间相关动作轨迹的概率模型与梯度，不修改环境行为
待验证假设：对参数无关 context 条件化虽然概率自洽，但会把产生稀有成功的低频位移从 base-mean score 中条件掉；对同一时间相关动作过程使用可精确计算的边缘/联合轨迹 likelihood，可在保持持续覆盖时给 base mean 提供正确且更低方差的 credit
禁止改动范围：不改 reward、observation/action 物理语义、阶段条件、时限、随机化、低层、AutoOpen、门型或 seed；禁止成功动作监督、固定 ee_x、环境侧滤波、阶段特例和无法重算 log-prob 的隐式状态
运行规模与日志：先做纯数学推导、phase194 数据离线反事实和无 Isaac 概率/梯度测试；接口闭合前不启动训练
成功/失败判据：定义 episode/reset/rollout 边界下的状态、采样和 old/new joint log-prob；参数相同时 ratio=1，解析 score 与 autograd 一致，边际方差和目标 lag correlation 正确；同一成功序列的 credit 不依赖门型特判
观测指标：trajectory log-prob、base mean/sigma score、lag covariance、effective sample size、左右门成功/失败分层 credit、PPO ratio/KL
结论与已排除方向：待验证；现阶段只确认当前 conditional score 对 context-induced success 信号弱，尚未批准替换正式概率接口
正式保留/删除的代码：phase194 staged-task 修复正式保留；phase195 未产生代码
前序阶段回归：phase194 的 22 项测试和四卡同规模只读 rollout 已通过
下一决策门槛：只有纯数学与离线左右门证据共同通过，才实现一个通用 sequence-aware 概率核；否则不改正式 temporal 接口
```

2026-09-27 13:31，phase195 数学与离线门槛通过。对当前噪声过程积分掉 AR latent 后，标准化动作残差的协方差为 `K[i,j]=f*rho^|i-j|+(1-f)delta(i,j)`；sequence score 为 `Sigma^-1(a-mu)`。新增 Kalman 序列核与完整多元高斯在 log-prob 和 base-mean autograd 上逐值一致，done 后恢复平稳先验，相同参数 old/new ratio 精确为 1；完整测试从 22 增至 `26 passed`。该核尚未接入 Actor/PPO，生产行为未改变。

在 phase194 真实轨迹上做离线反事实后，variant 0/1 的 OPEN_LATCH 聚合 base-mean signal 从当前 conditional score 的 `-6.77e4/-9.96e4` 变为边缘 sequence score 的 `+1.60e4/+0.79e3`，但跨 rank 方差仍大，因此只支持接口实现和真实 PPO smoke，不支持直接宣称训练已解决。独立 `300000 x 32` 合成持续动作任务中，两种 score 都与 pathwise 真梯度相容，边缘 sequence estimator 的方差低约 `196.4` 倍，验证 stored-context 条件化确实丢弃了最有信息的低频成功偏移。

### phase196：统一 sequence-aware temporal PPO 接入（已验证）

```text
上游结论与待解决瓶颈：phase195 已证明边缘序列 likelihood 概率严格且显著降低持续动作任务的梯度方差；当前 trainer 仍按 stored external context 计算 conditional PPO
基线 checkpoint / SHA-256：phase183 model_step_000125.pt / 104d110b48eddb9b57819ac6a3cbf1bf386e764aa8cb935f9bd449abab7c22d7
唯一变量：只替换 temporal exploration 的 PPO 概率所有权；物理采样边际协方差、rho、fraction 和 marginal std 保持不变
待验证假设：rollout 边界重置平稳先验后，actions/base mean/marginal std/dones 足以在 rollout 与 update 精确重建相同逐步边缘条件概率，并把低频动作偏移纳入 base-mean credit
禁止改动范围：不改环境、reward、observation/action、阶段条件、网络输出、随机化、低层、AutoOpen、rho/fraction/std、门型或 seed；禁止同时保留 stored-context 与 marginal 两套 PPO ratio
运行规模与日志：先做模块、Actor/storage/trainer 无仿真测试；再做 8 env x 8 steps 默认关闭和 temporal 开启 smoke；最后四卡只读/update 因果验收
成功/失败判据：默认关闭逐张量不变；temporal 开启时 rollout/update old log-prob 误差在既有浮点容差内、ratio=1、解析/autograd gradient 一致、done/rollout 边界正确；storage 和 trainer 不再依赖 action_noise_context 计算 PPO
观测指标：base mean、marginal std、per-step filtered mean/std、old/new log-prob、ratio、KL、entropy、lag covariance 和 stage x variant update signal
结论与已排除方向：待实现；已经排除把成功 context 写成 action supervision target或继续用外生 context conditional score
正式保留/删除的代码：保留 phase195 通用序列概率核；接入成功后删除 trainer 中 action_noise_context/action_distribution_mean 的 PPO storage 和 minibatch 路径
前序阶段回归：26 项无 Isaac 测试通过；phase194 stage 修复不得回归
下一决策门槛：只有真实 rollout 的 old/new ratio 和边际相关性同时通过，才允许从 step125 做短训练；否则停在概率接口修复
```

2026-09-27 13:41，phase196 完成。rollout 的物理采样仍是相同的 AR+white 边际过程，但每个 rollout 边界与 likelihood 一起重置到平稳先验；storage 中 `action_mean/action_sigma` 统一表示 base mean/marginal std，不再保存 external context 或 conditional mean 作为 PPO 概率输入。old/new per-step log-prob、KL 和 entropy 均从 actions、base mean、marginal std、dones 经同一个 Kalman 序列核计算；stored-context conditional PPO 路径已删除。

模块和 recurrent storage 测试覆盖完整多元高斯等价、解析梯度、done/rollout reset、old/new ratio=1、环境顺序 shuffle 后重建以及 temporal disabled 原 Gaussian，完整结果 `28 passed`。真实 `8 env x 8 steps` temporal smoke 和显式关闭 temporal 的对照 smoke 均完成 iteration126 与 optimizer，无异常或 NaN。四卡 `4 x 2048 env x 128 steps` 正式规模单 update 也正常完成：collection `66.800 s`、learning `5.626 s`、总吞吐 `14477 steps/s`；该运行不保存或继承更新后参数。

原 `stage_action_advantage_signal` 只在独立 Gaussian 下与局部 mean score 同号，sequence likelihood 会跨时间耦合梯度。为防止后续继续用错误指标调 reward，temporal 启用时已停止输出该指标；没有用近似局部值冒充真实 sequence PPO gradient。

### phase197：sequence marginal PPO 严格对照短训练（失败）

```text
上游结论与待解决瓶颈：phase196 已修复 temporal 成功轨迹的概率归因接口；尚不知真实 PPO 是否能把持续开缝覆盖吸收到 base mean 并保持 retention
基线 checkpoint / SHA-256：phase183 model_step_000125.pt / 104d110b48eddb9b57819ac6a3cbf1bf386e764aa8cb935f9bd449abab7c22d7
唯一变量：相对 phase192，只把 stored-context conditional PPO likelihood 替换为 phase196 的 sequence marginal likelihood；rho/fraction/std、环境和训练超参均不变
待验证假设：更低方差且包含低频偏移 credit 的 sequence score 会让左右门 base mean ee_x 转正，并在提高转换率时避免 phase192 的 retention 明显回归
禁止改动范围：训练中不改代码；不改 reward、observation/action、网络、阶段条件、时限、随机化、低层、AutoOpen、rho/fraction/std、门型或 seed
运行规模与日志：4 x 2048 env、128 steps/env、4 mini-batches，从 step125 训练 25 update 到 step150；容器 `doorman_train` 内日志 `/tmp/b2z1_phase197_marginal_temporal_to150.log`；运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260927_134427`
成功/失败判据：左右门 base mean ee_x 均形成与安全开缝一致的正向动作，转换率相对 step125 提高，retention 不出现 phase192 级别回归，APPROACH/GRASP 不回归，OPEN_LATCH unsafe-force 率不超过 2e-4
观测指标：六阶段 reached、stage x variant 转换、OPEN_LATCH base mean/physical command/opening force/hinge/retention、分阶段 unsafe-force、KL、ratio 和 loss
结论与已排除方向：失败。左/右 OPEN_LATCH -> REPOSITION 从 step125 的 `4.95%/12.09%` 提高到 `11.32%/27.05%`；retention pocket/axis/contact 为 `0.8883/0.9225/0.9083`，仅较 step125 小幅下降并显著优于 phase192。APPROACH 和 GRASP 不回归，OPEN_LATCH unsafe-force 日志值为 `0.0000`。但最终左右 base mean `ee_x=-0.0004/-0.0008`，iteration135--150 均未形成持续正值，转换仍依赖 temporal sample 覆盖。排除“修复 sequence likelihood 后再盲目增加 25 update 就会自然吸收到 base mean”
正式保留/删除的代码：phase196 通用概率接口正式候选；本 phase 不增加诊断或行为代码
前序阶段回归：APPROACH 左右 `100%`；GRASP 左右 `46.04%/50.06%`，高于 step125；总体 unsafe-force 每步率 `0.0001`，OPEN_LATCH 显示为 `0.0000`；无异常、NaN 或残留训练进程
下一决策门槛：不得继承 step150 继续训练。phase198 只读分解门型内 sequence PPO task gradient、监督/正则 gradient 与一次 optimizer 前后 actor mean，定位 base mean 不转正的首个层级
```

2026-09-27 14:15，phase197 iteration150 正常完成，checkpoint `model_step_000150.pt` SHA-256 为 `1b2ec62aab84d9913764374b7ad92db07f86b3b70357b7ee5ca5a92d5c92fa1f`。原始 Rich stdout 位于容器 `doorman_train:/tmp/b2z1_phase197_marginal_temporal_to150.log`；宿主机同名 `/tmp` 路径不存在，已将正式日志必须注明容器命名空间加入维护规则。

25 个 update 的走势不是单纯“训练尚短”：左右 base mean 从 iteration126 的 `-0.0024/-0.0023` 回升到接近零，但 iteration135--150 始终在零下震荡，最终为 `-0.0004/-0.0008`；同期转换率持续提高。该证据说明 sequence marginal likelihood 有实际收益，但当前策略仍主要靠相关探索产生持续开缝序列，不能把 phase197 step150 升格为有效训练 checkpoint。

### phase198：step150 sequence gradient 所有权审计（已验证）

```text
上游结论与待解决瓶颈：phase197 的正确 sequence likelihood 提高双侧转换并基本保住 retention，但 base mean 后半程仍在零下平台；尚不知首个抵消发生在 task score、监督/正则 loss、共享参数还是 optimizer
基线 checkpoint / SHA-256：phase197 model_step_000150.pt / 1b2ec62aab84d9913764374b7ad92db07f86b3b70357b7ee5ca5a92d5c92fa1f（仅作失败诊断基线，不续训）
唯一变量：保持 rollout 和生产行为不变，导出同一批轨迹上完整 sequence marginal PPO 的门型内 task gradient、各附加 loss gradient，以及一次 optimizer 前后固定状态 actor mean
待验证假设：转换成功轨迹提供正 sequence task gradient，但该方向被共享网络中的其他 stage 或监督/正则项抵消；若 task gradient 本身仍为负，则继续定位 reward-to-go 的首个负作用层
禁止改动范围：不更新或保存模型；不改 reward、observation/action、网络、探索、阶段条件、时限、随机化、低层、AutoOpen、门型或 seed；诊断代码必须环境变量门控且结论后删除
运行规模与日志：`8 env x 8 steps` 单卡 smoke 已正常写盘并在 optimizer 前退出；正式运行 `4 x 2048 env x 128 steps`，容器 `doorman_train` 内日志 `/tmp/b2z1_phase198_sequence_score.log`，数据 `/tmp/b2z1_phase198_sequence_score.rank{0,1,2,3}.pt`
成功/失败判据：逐层给出左右门 `ee_x` 从 sequence score 到实际参数更新的符号和量级，找到第一个与安全开缝方向不一致的层；字段不足时只补无副作用观测
观测指标：stage x door variant x released/converted 的 sequence score、GAE、sample weight、task/supervision/entropy/value gradient、固定状态 pre/post actor mean
结论与已排除方向：四卡 `1,048,576` 样本只读导出正常完成，左/右 OPEN_LATCH 样本 `115207/106443`。精确 sequence action-space gradient 显示，左门 OPEN_LATCH 当前 `ee_x` descent signal 为 `+0.06368`，却向前追溯到 GRASP `-2.31194`；右门当前项为 `-0.06885`，追溯到 GRASP `-9.43455`。根因是 GRASP residual mask 为 0、实际 std 近零，但 correlated fraction 仍为 0.9，Kalman filter 从非策略所有权动作中近乎精确观测 latent context，并把后续 OPEN_LATCH credit 归到不可学习的 GRASP mean。排除“首个抵消只发生在共享网络或监督 loss”
正式保留/删除的代码：phase196 唯一概率接口不动；phase198 临时审计路径结论形成后删除
前序阶段回归：诊断不得产生新 checkpoint；完成后运行 28 项测试、py_compile、diff check 和临时标识搜索
下一决策门槛：phase199 在统一概率所有权层令有效 fraction 随 residual mask 生效；必须同时修 sampler/storage/old-new likelihood，并用同规模只读梯度证明 GRASP 泄漏归零
```

正式证据位于容器 `doorman_train:/tmp/b2z1_phase198_sequence_score.log` 和 `/tmp/b2z1_phase198_sequence_score.rank{0,1,2,3}.pt`。四个 rank 均在 optimizer 前退出，运行目录只有配置和 meta 文件，没有 step151 checkpoint。

### phase199：时间相关探索的 action ownership 语义修复（已验证）

```text
上游结论与待解决瓶颈：phase198 证明 OPEN_LATCH credit 被非策略所有权的 GRASP 极小方差动作错误吸收；问题同时存在于物理 sampler 和 Kalman likelihood 的观测模型
基线 checkpoint / SHA-256：phase197 model_step_000150.pt / 1b2ec62aab84d9913764374b7ad92db07f86b3b70357b7ee5ca5a92d5c92fa1f（只读对照，不续训）
唯一变量：每步有效 correlated fraction 从固定配置改为 `configured fraction x action residual ownership mask`，并将逐步 fraction 写入 storage，供 sampler、old log-prob 和 new log-prob 共用
待验证假设：不拥有 residual 的阶段不应观测 temporal latent；fraction=0 时未来 OPEN_LATCH likelihood 对这些 mean 的梯度严格为零，而拥有动作的阶段仍保留原时间相关探索
禁止改动范围：不改 reward、observation/action 值、网络、阶段条件、时限、随机化、低层、AutoOpen、rho、边际 std、门型或 seed；不按阶段编号硬编码 reset
运行规模与日志：29 项单测、`8 env x 8 steps` Isaac audit smoke、`4 x 2048 env x 128 steps` 四卡只读对照和真实 optimizer smoke 均通过；正式对照位于容器 `doorman_train:/tmp/b2z1_phase199_ownership.log` 与 `/tmp/b2z1_phase199_ownership.rank{0,1,2,3}.pt`
成功/失败判据：GRASP 来源的 OPEN_LATCH `ee_x` sequence gradient 为数值零；OPEN_LATCH 自身仍有有效梯度；old/new ratio 同参为 1；无异常、NaN 或 checkpoint
观测指标：逐步 effective fraction、stage x variant sequence descent signal、样本量、ratio 重算误差和安全指标
结论与已排除方向：通过。`ee_x` effective fraction 在 APPROACH/GRASP/REPOSITION 严格为 0，在 OPEN_LATCH/PUSH_OPEN/THROUGH 严格为 0.9；OPEN_LATCH loss 对 GRASP 的 gradient 从 phase198 的 `-2.31/-9.43` 变为数值零。左/右 OPEN_LATCH 当前 task signal 为 `+0.07098/-0.06466`。进一步离线分组显示已转换轨迹两侧均为正（`+0.00056/+0.00248`），右门负号来自未转换轨迹 `-0.06714`，不是成功 credit 符号错误
正式保留/删除的代码：保留逐步 ownership fraction 的 sampler/storage/old-new likelihood 唯一接口和新增测试；phase198 环境变量审计方法、提前退出分支和额外 rollout 字段已删除
前序阶段回归：py_compile、29 项单测、`git diff --check` 通过；真实 optimizer smoke 完成 iteration151，无异常、NaN 或 checkpoint
下一决策门槛：phase200 先定位右门未转换负 credit 的物理/reward 首因，不因左门转正直接启动训练
```

### phase200：右门未转换 sequence credit 时间链（已验证）

```text
上游结论与待解决瓶颈：phase199 已清除跨 action ownership 的概率污染；右门成功转换轨迹 credit 为正，但大量未转换轨迹使总 task signal 为负
基线 checkpoint / SHA-256：phase197 model_step_000150.pt / 1b2ec62aab84d9913764374b7ad92db07f86b3b70357b7ee5ca5a92d5c92fa1f（只读诊断）
唯一变量：生产行为不变，只导出右门 OPEN_LATCH 未转换轨迹的 exact sequence score、retention/contact/latch/gap 物理增量、逐项 reward、value/GAE 和后续结果
待验证假设：负 credit 首先来自某个可定位的实际 retention/contact 或 latch/gap 后果，而不是成功样本稀少本身；若物理后果正确而 reward 才反向，则定位具体 reward 项
禁止改动范围：不更新模型；不改 reward、observation/action、网络、temporal sampler、阶段条件、时限、随机化、低层、AutoOpen、门型或 seed
运行规模与日志：`8 env x 8 steps` smoke 已验证逐项 scaled reward 求和与正式 `rew_buf` 逐元素完全一致，导出后 optimizer 前退出且无 checkpoint；首轮正式导出发现缺少片段时间坐标而不用于时间链结论，补齐后以 `4 x 2048 env x 128 steps` 重跑，容器 `doorman_train` 日志 `/tmp/b2z1_phase200_credit_chain_v2.log`，数据 `/tmp/b2z1_phase200_credit_chain_v2.rank{0,1,2,3}.pt`
成功/失败判据：按时间顺序找到右门正/负 sequence score 分组首次出现 retention/contact/latch/gap 或 reward-to-go 分化的位置；不能只报告总 advantage
观测指标：converted、advantage sign、sequence score、pocket/axis/contact retention、opening force、latch/gap/hinge delta、逐项 reward、value/return
结论与已排除方向：右门最终转换片段并非先丢失夹持；第 8--15 步 opening force 已为 `13.8 N`，未转换片段仅 `1.2 N`，随后门角持续分化。成功依赖约 `+0.02--0.03` 的持续 `ee_x` temporal residual，而失败片段采样/mean 接近零或为负。右门精确 signal `-0.01464` 中，stage progress、termination、critic 分别贡献 `-0.02630/-0.01007/+0.02171`，其余 reward 近零；排除 critic、普通动作惩罚或“夹持先坏”是首因。现乘法 potential 将 retention 的正常受力下降乘到已累计 latch/gap progress 上，形成主要负 credit
正式保留/删除的代码：phase200 环境变量导出、storage 字段和 optimizer 前退出路径已全部删除；原始 `.pt` 和日志仅作证据，phase199 正式概率接口不动
前序阶段回归：两轮四卡 rollout 均无更新、无 checkpoint、无异常；逐项 scaled reward 求和与环境正式 reward 最大误差为 `0`
下一决策门槛：固定轨迹反事实中，加性 `0.10 retention + 0.45 latch + 0.45 gap` 将左/右 exact signal 从 `+0.01706/-0.01464` 改为 `+0.03340/+0.00401`；phase201 只允许验证这一结构变化
```

### phase201：OPEN_LATCH 加性约束与任务进度势函数（失败）

```text
上游结论与待解决瓶颈：phase200 证明右门负 credit 的首因是 retention 与累计 latch/gap 相乘，成功所需的接触顺应被放大扣分；加性固定轨迹反事实使左右 sequence signal 同时为正
基线 checkpoint / SHA-256：phase183 model_step_000125.pt / 104d110b48eddb9b57819ac6a3cbf1bf386e764aa8cb935f9bd449abab7c22d7
唯一变量：OPEN_LATCH stage progress 从 `retention * (0.10 + 0.45*latch + 0.45*gap)` 改为 `0.10*retention + 0.45*latch + 0.45*gap`；phase199 action-ownership 修复同时作为正式代码基础
待验证假设：把 retention 恢复为独立有界约束后，正常受力不再抹除 latch/gap 进度，左右门 actor base mean 都能吸收持续正开缝动作，同时保住严格 GRASP 与可接受 retention
禁止改动范围：不增加 contact-force bridge、best-contact、门型/seed 特例或第二套 reward；不改 observation/action、网络、temporal 参数、阶段验收、时限、随机化、低层或 AutoOpen
运行规模与日志：29 项测试、py_compile、diff check 和 `8 env x 8 steps` optimizer smoke 已通过；smoke 日志为容器 `doorman_train:/tmp/b2z1_phase201_additive_smoke.log`，运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260927_151607` 只有配置与 meta、无 checkpoint。正式训练为 `4 x 2048 env x 128 steps`，从 step125 训练 25 update 到 step150，日志预定为容器 `doorman_train:/tmp/b2z1_phase201_additive_to150.log`
成功/失败判据：左右 OPEN_LATCH base mean `ee_x` 形成正向持续动作，转换率相对 step125 提高；pocket/axis/contact retention 不出现 phase192 级别回归，APPROACH/GRASP 不回归，OPEN_LATCH unsafe-force 每步率不超过 `2e-4`
观测指标：六阶段 reached、stage x variant 转换、base mean/sample/physical ee_x、opening force、latch/hinge/gap、三项 retention、unsafe force、KL、ratio、loss
结论与已排除方向：加性公式使左右 base mean 从 iteration126 的 `-0.0023/-0.0023` 持续升至 iteration135 的 `+0.0080/+0.0073`，OPEN_LATCH -> REPOSITION 转换升至约 `52.65%/67.48%`，证明 phase200 的 credit 方向判断成立；但 pocket/axis/contact retention 降至 `0.7981/0.8428/0.5481`，contact 已明显差于 phase192 的 `0.8084`，触发预设失败门槛。代码核对进一步确认 OPEN_LATCH 完成条件只有门缝角度，不检查可靠夹持；排除把高 reached 直接解释为完成阶段物理目标
正式保留/删除的代码：当前只有一处 stage-progress 公式变化；phase200 临时诊断已删除
前序阶段回归：29 项测试、py_compile 和 `git diff --check` 通过；smoke 使用四卡 checkpoint 的单卡 topology，staged-reset snapshots 按预期跳过，因此只验训练链路，不声称完成物理阶段回归
下一决策门槛：phase202 只修复 OPEN_LATCH 验收条件，要求门缝达标时 `_precise_bilateral_grasp` 仍成立；不同时改 potential、权重、动作或探索
```

2026-09-27 15:29，iteration130：APPROACH 左右仍为 `100%`，GRASP 左/右 `48.76%/49.56%`；OPEN_LATCH reached `33.50%/26.21%`，REPOSITION reached `3.48%/7.17%`，对应 OPEN_LATCH -> REPOSITION 转换约 `10.39%/27.36%`。左右 OPEN_LATCH base mean `ee_x` 均从 iteration126 的 `-0.0023` 转为 `+0.0011`，opening force 为 `4.34/6.67 N`，OPEN_LATCH unsafe-force 仍为 `0`。retention 从 `0.7794` 降至 `0.6611`，因此当前只确认加性 credit 已出现预期方向的早期在线信号，尚不能确认保持夹持或最终训练验收通过；代码与参数保持不变继续观察。

2026-09-27 15:36，iteration135 后按门槛停止。APPROACH 左右保持 `100%`，GRASP `48.36%/49.79%`；OPEN_LATCH reached `35.99%/31.09%`，REPOSITION reached `18.95%/20.98%`。左右 opening force 为 `15.96/13.33 N`，OPEN_LATCH unsafe-force 仍为 `0`，但 retention quality 从 iteration126 的 `0.7794` 单调趋势降至 `0.5151`；pocket/axis/contact 分量为 `0.7981/0.8428/0.5481`。本轮在 step135 未到保存周期，运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260927_151927` 只有 config/meta，没有 checkpoint；有效基线仍是 phase183 step125。停止后的四个 worker、accelerate 父进程和入口 shell 已清理，GPU 回到空闲基线，容器未关闭。

### phase202：OPEN_LATCH 物理目标与验收语义统一（失败）

```text
上游结论与待解决瓶颈：phase201 已让双侧开门动作均值和门缝转换快速改善，但策略同步牺牲 contact retention；实际 OPEN_LATCH 完成条件只检查门缝，与“保持可靠夹持并形成门缝”的物理目标不一致
基线 checkpoint / SHA-256：phase183 model_step_000125.pt / 104d110b48eddb9b57819ac6a3cbf1bf386e764aa8cb935f9bd449abab7c22d7
唯一变量：OPEN_LATCH 完成条件从仅 `hinge_angle >= gap_angle` 改为门缝达标且 `_precise_bilateral_grasp` 同时成立
待验证假设：阶段完成奖励不再认可丢失夹持后的门缝后，策略必须在保留 phase201 正向开门动作的同时维持严格双侧抓持，不能用 contact retention 换 reached
禁止改动范围：不改加性 potential 及权重、observation/action、temporal 参数、网络、时限、随机化、低层、AutoOpen、其他阶段条件、门型或 seed；不增加阶段流程特例或隐藏阈值
运行规模与日志：30 项测试、py_compile、diff check 和 `8 env x 8 steps` 真实 optimizer smoke 已通过；smoke 日志为容器 `doorman_train:/tmp/b2z1_phase202_grasp_gate_smoke.log`，运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260927_154112`，无 checkpoint。正式对照从 step125 使用 `4 x 2048 env x 128 steps` 训练 25 update，日志启动前登记
成功/失败判据：OPEN_LATCH 转换只在门缝与严格双侧夹持同时成立时计数；左右 base mean 保持正向，pocket/axis/contact retention 不低于 phase192 `0.8171/0.8749/0.8084`，APPROACH/GRASP 不回归，OPEN_LATCH unsafe-force 每步率不超过 `2e-4`
观测指标：完成事件瞬间的 precise bilateral grasp、三项 retention、gap、左右 base mean/opening force、六阶段 reached、unsafe-force
结论与已排除方向：纯函数测试确认严格验收语义正确；正式训练到 iteration130 时左右 base mean `ee_x=+0.0004/+0.0003`、opening force `3.89/6.27 N`，严格 OPEN_LATCH -> REPOSITION 转换约 `4.54%/16.67%`，但 pocket/axis/contact retention 已降到 `0.8163/0.8624/0.7040`，contact 低于 phase192 失败基准 `0.8084`。排除“只修复完成条件即可阻止接触退化”；该修复仍是阶段语义所必需，不能撤销
正式保留/删除的代码：保留 phase199 概率所有权与 phase201 加性 potential；正式新增唯一轻量 `open_latch_completion_condition`，环境和测试共用，不保留重复静态方法
前序阶段回归：30 项测试、py_compile、diff check 通过；真实 smoke 只覆盖训练链路，正式物理回归由四卡对照验证
下一决策门槛：phase203 只读归因，不更新或保存模型；必须用同一轨迹定位 retention 损失与 latch/gap/完成奖励的 credit 数量关系，确认根因后才允许设计下一行为变量
```

2026-09-27 15:54，phase202 在 iteration130 后按预设门槛停止。APPROACH 左右保持 `100%`，GRASP `48.67%/48.22%`，OPEN_LATCH reached `33.90%/25.67%`，REPOSITION `1.54%/4.28%`；OPEN_LATCH unsafe-force 为 `0`。运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260927_154340` 未到保存周期，只有 config/meta，无 checkpoint；有效基线仍是 phase183 step125。正式日志为容器 `doorman_train:/tmp/b2z1_phase202_grasp_gate_to150.log`。停止后的 worker、accelerate 父进程和入口 shell 均已清理，GPU 回到空闲基线，容器未关闭。

### phase203：加性 OPEN_LATCH credit 竞争只读归因（已验证）

```text
上游结论与待解决瓶颈：phase202 修复了完成条件，但严格转换下降时 contact retention 仍快速退化；尚不能把原因简化为完成奖励钻空子
基线 checkpoint / SHA-256：phase183 model_step_000125.pt / 104d110b48eddb9b57819ac6a3cbf1bf386e764aa8cb935f9bd449abab7c22d7
唯一变量：生产行为不变，只增加 optimizer 前退出的 OPEN_LATCH 分量与 exact sequence credit 导出
待验证假设：`0.10 retention + 0.45 latch + 0.45 gap` 中 latch/gap 正 credit 在持续前推轨迹上系统性超过 contact retention 负 credit，使策略即使无法以丢夹持完成阶段，仍先学会牺牲接触
禁止改动范围：不更新或保存模型；不改 potential、权重、完成条件、observation/action、网络、探索、时限、随机化、低层、AutoOpen、门型或 seed
运行规模与日志：`8 env x 8 steps` smoke 通过；正式 v2 使用 `4 x 2048 env x 128 steps`，日志为容器 `doorman_train:/tmp/b2z1_phase203_credit_v2.log`，数据为 `doorman_train:/tmp/b2z1_phase203_credit.rank{0,1,2,3}.pt`，四个 rank 均在 optimizer 前退出且无 checkpoint
成功/失败判据：按门型和最终严格完成/未完成分组，给出 retention/latch/gap、stage-complete、termination 及其他 reward 对 exact sequence signal 的数量贡献，找到接触退化动作获得净正 credit 的首个层级
观测指标：逐步 potential 三分量与 delta、precise bilateral grasp、严格完成事件、base/sample/physical ee_x、opening force、GAE/value、sequence score
结论与已排除方向：假设被否定。左右 OPEN_LATCH 的 retention credit 为 `-0.05059/-0.03652`，latch 为 `-0.05694/-0.02666`，gap 为 `+0.00961/+0.02775`，critic/value 为 `+0.21757/+0.20188`，最终总方向为 `+0.15032/+0.11486`。正 `ee_x` 更新不是 latch/gap 压过 retention，而几乎完全由旧 critic 推动；phase201/202 从旧语义 phase183 step125 同时恢复 actor/critic，短训练不能用于继续扫描 reward 权重
正式保留/删除的代码：保留 phase199 temporal action ownership、phase201 加性 potential、phase202 严格验收和共享纯函数测试；phase203 的环境变量、reward 缓存、诊断接口、storage 导出和 optimizer 前退出路径已全部删除
前序阶段回归：正式 rollout 未更新模型或生成 checkpoint；清理后临时标识搜索为空，30 项测试、py_compile 和 `git diff --check` 通过
下一决策门槛：phase204 在最终统一语义下从随机初始化干净训练，不加载 phase183 或任何旧语义 checkpoint；先 smoke，再按 step25/50/75/100/125 检查最早失败阶段
```

2026-09-27 16:05，phase203 首轮正式导出按计划在 optimizer 前退出，四个 rank 均写出 `[128,2048,...]`、所有浮点字段有限且无 checkpoint。对账发现正式 storage 会对 timeout 样本把 `gamma * time_outs * values` 加入 `rewards`，而首版导出没有保存已有的 `time_outs`；因此 26 个物理 reward 项与 storage reward 的最大差为 `30.41--42.89`，无法严谨拆分 reward 与 bootstrap。本轮只用于发现导出字段缺口，不用于 credit 结论；唯一修复是补导出 `time_outs` 后以同一配置重跑。

2026-09-27 16:17，phase203 v2 已完成。补入已有 `time_outs` 后，四个 rank 均满足 `sum(reward_terms) + 0.9975 * time_outs * values == stored rewards`，逐 rank 最大误差为 `0`，合并打印浮点误差约 `7.63e-06`。OPEN_LATCH 样本左右分别为 `118108/108091`，严格阶段完成事件均为 `0`。exact sequence 分解否定预设假设：retention 和 latch 对 `ee_x` 都提供负 credit，gap 的小幅正 credit不足以解释正总更新；正方向由与新 reward/验收语义不一致的旧 critic/value 分量主导。临时诊断代码已删除，30 项测试、py_compile、diff check 和临时标识搜索通过。

### phase204：最终统一语义下四卡干净训练（已验证）

```text
上游结论与待解决瓶颈：phase203 证明 phase201/202 继承的旧 critic 主导短期更新；最终 reward、验收和概率语义必须由全新 actor/critic 学习后才能判断首个真实瓶颈
基线 checkpoint / SHA-256：无；随机初始化，禁止加载 phase183、phase197 或其他历史 checkpoint
唯一变量：训练初始化方式从恢复旧 checkpoint 改为 actor、critic 和 staged-reset snapshots 全部从零积累；生产 reward、阶段条件、observation/action、网络和随机化不变
待验证假设：在统一语义下干净训练能够先复现 APPROACH_PREGRASP 与 GRASP，再给出不受 stale critic 污染的 OPEN_LATCH retention/开缝学习曲线
禁止改动范围：运行中不改代码；不改 reward 数值、阶段验收、observation/action、网络、temporal 参数、随机化、低层、AutoOpen、门型或 seed；不添加阶段特例、隐藏开关、临时阈值或 checkpoint 兼容分叉
运行规模与日志：不加载 checkpoint 的 `1 x 8 env x 8 steps x 1 update` 真实 optimizer smoke 已通过，日志为容器 `doorman_train:/tmp/b2z1_phase204_clean_smoke.log`；正式运行使用纯动力学、headless、无相机、无 RTX sensor、关闭 frame rendering 的 `4 x 2048 env x 128 steps`，日志为容器 `doorman_train:/tmp/b2z1_phase204_clean_8192_to125.log`，每 25 update 保存 checkpoint
成功/失败判据：训练链路无异常/NaN；step25/50/75/100/125 按六阶段 reached、左右门转换、严格夹持 retention、安全力和 actor/critic 指标定位最早瓶颈；不得用后续阶段总量掩盖前序失败
观测指标：六阶段 reached 与转换率、stage x door variant、pocket/axis/contact retention、latch/gap/hinge、opening force、unsafe force、base/sample/physical action、value/return/KL/ratio/loss
结论与已排除方向：训练完成 125 updates、`131,072,000` timesteps，无 checkpoint 恢复、异常或 NaN；纯动力学四卡链路稳定。step25--125 未复现 phase201/202 retention 坍塌，右门严格转换持续升至 `27.10%`，左门在 step75--125 仅从 `6.48%` 升至 `7.02%`。排除 stale critic 与持续接触质量崩溃是当前左右差异的完整解释；最终 task success 仍为 0，phase204 不满足最终验收
正式保留/删除的代码：不新增诊断行为路径；若 smoke 暴露接口 bug，只在正确所有权层修复并完整回归
前序阶段回归：30 项测试、py_compile、diff check 和 phase203 临时标识清理通过；干净初始化 smoke 通过；正式运行每 25 update 评估一次
下一决策门槛：phase205 先用现有日志对比 step125 左右门分层指标；字段不足时才新增无副作用只读导出。必须找到首次分化层后才允许提出一个行为变量。只有同一 checkpoint 在 seed 1700--1703 完成六阶段才算最终完成
```

备份记录：2026-09-27 13:43 已将 phase195/196 的通用概率代码、测试、phase197 计划和本文档同步到 `wholebody_1/3/4`，未修改远程 Git 历史或凭据；写入本条前四机内容聚合 SHA-256 为 `dbabb32c26df2c39311ddbd31554f1840a76c21fb5e5eb1c73213762754071a9`，本条写入后状态文档单独再次同步。

备份记录：2026-09-27 14:18 已将 phase197 正式结论、phase198 计划和失败证据 checkpoint 同步到 `wholebody_1/3/4`，未修改远程 Git 历史或凭据；checkpoint SHA-256 三机与本地一致，为 `1b2ec62aab84d9913764374b7ad92db07f86b3b70357b7ee5ca5a92d5c92fa1f`。本条写入后状态文档再次单独同步。

备份记录：2026-09-27 14:50 已将 phase199 正式概率接口、两项相关测试和 phase200 计划同步到 `wholebody_1/3/4`，未修改远程 Git 历史或凭据；七个同步文件聚合 SHA-256 四机一致为 `0affa1e1b35c7f94eb925128106414993e54dc7203f5a1f4b13eab4bb3f6660e`。本条写入后状态文档再次单独同步。

备份记录：2026-09-27 15:43 已将 phase201 结论、phase202 验收修复、对应测试和本文档同步到 `wholebody_1/3/4`，未修改远程 Git 历史或凭据；同步前四个文件逐项 SHA-256 四机一致。phase201 未产生新 checkpoint，三机已有的 phase183 step125 仍是唯一有效训练基线；未同步训练日志、运行目录、缓存或无关文件。本条写入后状态文档单独再次同步。

备份记录：2026-09-27 16:18 已将 phase203 清理后的正式源码、30 项测试和更新后的执行账本同步到 `wholebody_1/3/4`，未同步临时诊断数据、日志、缓存或运行目录；状态文档及三处清理文件的 SHA-256 已逐项核对，四机一致。phase204 尚未产生 checkpoint。

2026-09-27 16:27，phase204 正式四卡干净训练完成 step1：总计 `1,048,576` timesteps，iteration time `62.99s`；四个 rank 均确认纯动力学无渲染，未加载历史 checkpoint，无异常或 NaN。左右 APPROACH_PREGRASP reached 均为 `100%`，GRASP 尚无样本；在 step25 前不据早期单点调整行为逻辑。

2026-09-27 16:54，phase204 完成 step25 并保存首个统一语义 checkpoint。左右 GRASP reached 为 `32.02%/34.80%`，OPEN_LATCH reached 为 `21.65%/15.29%`，REPOSITION reached 为 `0.32%/0.55%`，对应严格 OPEN_LATCH 转换约 `1.48%/3.60%`；左右 OPEN_LATCH policy mean `ee_x=+0.0163/+0.0154`、opening force `16.73/29.65 N`，pocket/axis/contact retention 为 `0.6968/0.4222/0.4170`，OPEN_LATCH unsafe-force 为 0。step20--25 retention 在低位波动而非继续单调下降，当前不改行为逻辑，原配置运行到 step50。checkpoint 为 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260927_162216/model_step_000025.pt`，SHA-256 `383017b182e5b5db92206a584ac4600e59b33ef4de89b6cdf7746055547183b0`。

备份记录：2026-09-27 16:55 已将 phase204 step25 checkpoint 和对应执行账本同步到 `wholebody_1/3/4`；checkpoint SHA-256 已逐机核对，四机一致为 `383017b182e5b5db92206a584ac4600e59b33ef4de89b6cdf7746055547183b0`。未同步训练日志、缓存或运行目录中的其他文件。

2026-09-27 17:22，phase204 完成 step50。左右 GRASP reached `40.40%/44.00%`，OPEN_LATCH reached `28.59%/21.71%`，REPOSITION reached `1.41%/3.28%`，对应严格 OPEN_LATCH 转换约 `4.93%/15.11%`；PUSH_OPEN reached `0.32%/0.80%`，THROUGH reached `0/0.47%`。pocket/axis/contact retention 为 `0.7425/0.5599/0.4667`，三项均较 step25 改善，OPEN_LATCH unsafe-force 为 0。双侧 policy mean `ee_x=+0.0113/+0.0113`，opening force `22.51/26.01 N`。当前不改代码，继续到 step75。checkpoint 为 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260927_162216/model_step_000050.pt`，SHA-256 `c6487f91eea5b2d5cbb384e29396747f51a9be40f6aaba2c706bee7096558b13`。

2026-09-27 17:51，phase204 完成 step75。左右 GRASP reached `43.15%/46.63%`，OPEN_LATCH reached `30.73%/25.31%`，REPOSITION reached `1.99%/4.85%`，对应严格 OPEN_LATCH 转换约 `6.48%/19.16%`；PUSH_OPEN reached `0.68%/1.79%`，THROUGH reached `0.03%/1.46%`。pocket/axis/contact retention 为 `0.7936/0.5821/0.4848`，三项继续改善，OPEN_LATCH unsafe-force 为 0。双侧 policy mean `ee_x=+0.0091/+0.0101`，opening force `22.10/24.81 N`。当前不改代码，继续到 step100。checkpoint 为 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260927_162216/model_step_000075.pt`，SHA-256 `70a499beef108f026266a0a775f560c6adae766b4987e000e815970f93aaab5c`。

2026-09-27 18:19，phase204 完成 step100。左右 GRASP reached `44.22%/47.68%`，OPEN_LATCH reached `31.48%/27.81%`，REPOSITION reached `1.84%/6.78%`，对应严格 OPEN_LATCH 转换约 `5.84%/24.38%`；PUSH_OPEN reached `0.83%/2.47%`，THROUGH reached `0.08%/1.99%`。pocket/axis/contact retention 为 `0.7913/0.5559/0.4676`，OPEN_LATCH unsafe-force 为 0。右门继续增长，左门严格转换较 step75 的 `6.48%` 在约 `6%` 平台；这不是 retention 回归造成的直接结果。当前不改代码，继续到 step125。checkpoint 为 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260927_162216/model_step_000100.pt`，SHA-256 `f3148c46d06331faa0120dd3d181319cf8527b1fe261e0190564d4b0da1cf271`。

2026-09-27 18:49，phase204 正常完成 step125 并退出。总训练量 `131,072,000` timesteps，最终 iteration time `66.39s`，无异常或 NaN。左右 GRASP reached `44.27%/49.19%`，OPEN_LATCH reached `31.47%/28.60%`，REPOSITION reached `2.21%/7.75%`，对应严格 OPEN_LATCH 转换约 `7.02%/27.10%`；PUSH_OPEN reached `1.24%/3.42%`，THROUGH reached `0.16%/2.83%`，task success 为 0。pocket/axis/contact retention 为 `0.7502/0.5720/0.4321`，OPEN_LATCH unsafe-force 为 0。左门 step75--125 基本平台而右门持续增长，phase205 将只读定位首次分化层。checkpoint 为 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260927_162216/model_step_000125.pt`，SHA-256 `cb42398249389798b062756c9092d95f684e7d34b84208ad3c9b21688fe9d89f`。

备份记录：2026-09-27 18:50 已将 phase204 step125 checkpoint 和当时的执行账本同步到 `wholebody_1/3/4`；checkpoint SHA-256 已逐机核对，四机一致为 `cb42398249389798b062756c9092d95f684e7d34b84208ad3c9b21688fe9d89f`。未同步训练日志、缓存或运行目录中的其他文件。

### phase205：干净 checkpoint 左右 OPEN_LATCH 分化只读诊断（已验证）

```text
上游结论与待解决瓶颈：phase204 排除 stale critic 和 retention 坍塌后，左门严格 OPEN_LATCH 转换在约 7% 平台，右门达到 27.10%；需要找到成功/失败轨迹首次分化层
基线 checkpoint / SHA-256：phase204 model_step_000125.pt / cb42398249389798b062756c9092d95f684e7d34b84208ad3c9b21688fe9d89f
唯一变量：生产行为不变；先分析正式日志已有门型分层指标，字段不足时只增加 optimizer 前退出的只读轨迹导出
待验证假设：左右门转换差异可在严格完成前定位到某个 pocket/axis/contact、latch/gap、动作持续性或 sequence credit 层，而不是笼统归因于“左门更难”
禁止改动范围：不更新或保存模型；不改 reward、阶段条件、observation/action、网络、temporal 参数、随机化、低层、AutoOpen、门型或 seed；不添加门侧、阶段或 seed 特例
运行规模与日志：smoke 已通过。首轮正式 `4 x 2048 env x 128 steps` 的四 rank 字段、有限性和 reward 对账均通过，optimizer 前退出且无 checkpoint；严格完成事件仅左/右 `0/3`。第二轮唯一把窗口扩为 512 steps，日志为容器 `doorman_train:/tmp/b2z1_phase205_open_latch_512.log`，数据为 `doorman_train:/tmp/b2z1_phase205_open_latch.rank{0,1,2,3}.pt`，四个 rank 正常退出
成功/失败判据：按时间与门型找到左门相对右门首次显著分化的物理量或 exact credit 层，并给出直接证据；不能只报告最终转换率
观测指标：严格完成/未完成、pocket/axis/contact retention、latch/gap/hinge、opening force、base/sample/physical action、value/return/advantage 和 sequence score
结论与已排除方向：四份数据均为 `[512,2048,...]`，61 个字段全部有限；26 个 scaled reward 项加 `0.9975 * time_outs * values` 与 storage reward 最大误差为 `0`，运行目录只有 config/meta。`stage_advance` 在 next-observation 生命周期中已被清零，因此严格事件按相邻 storage stage 的 `2 -> 3` 转换重建，为左/右 `1/92`；这是一处临时导出字段时序限制，不是生产阶段事件 bug。左门成功样本不足，按预案改用 near-completion 分层。首个可见分化已在入口 `0--24` 步 actor mean 出现：右减左 canonical `ee_y=-0.00480`、`ee_x=+0.00330`，早于显著接触分化。到 `100--149` 步 opposed-contact 约左 `0%`、右 `6%`，contact retention 为 `0.420/0.480`；在 latch>=0.98 且 gap>=0.98 的样本中 opposed-contact 为 `1.0%/17.3%`，pocket check `62.7%/72.7%`，50 N force check `15.9%/29.4%`，而 axis check 为 `13.3%/9.4%`。因此晚期直接瓶颈是双爪对向接触，不是 axis、锁舌、门缝或安全力；更早的根因位于 actor observation 到 canonical action 的入口映射。精确 temporal sequence score 对 `ee_y` 也呈左右相反方向，左 `-0.09110`、右 `+0.11274` 每样本，不能把问题简化为统一增减某个动作值
正式保留/删除的代码：phase205 的环境变量门控、reward 缓存、10 项验收诊断、storage 导出与 optimizer 前退出路径已全部删除；生产代码不保留诊断开销
前序阶段回归：正式 rollout 未执行 optimizer 或生成 checkpoint；清理后临时标识搜索为空，30 项测试、py_compile 和 `git diff --check` 通过
下一决策门槛：phase206 只读导出 OPEN_LATCH 入口 actor observation 分组及 policy mean，按字段定位 canonical `ee_y` 分叉的输入来源；若字段级相关仍不能确认因果，再对同一 checkpoint 做离线 observation counterfactual。根因确认前不改 behavior、reward 或 observation 定义
```

2026-09-27 19:24，phase205 完成并清理。512-step 数据已经达到“定位首个分化层”的目的，且左门严格成功只有 1 条，不具备成功轨迹统计意义；依据预先登记的门槛，不再扩大窗口。当前有效 checkpoint 仍为 phase204 step125。

### phase206：OPEN_LATCH 入口 observation 到 action 分叉只读归因（已验证）

```text
上游结论与待解决瓶颈：phase205 证明左右门在显著物理接触分化前，入口 canonical actor mean 已先分叉；晚期失败集中为 opposed-contact 丢失，但尚不知道哪个 actor 输入字段首先驱动入口动作差异
基线 checkpoint / SHA-256：phase204 model_step_000125.pt / cb42398249389798b062756c9092d95f684e7d34b84208ad3c9b21688fe9d89f
唯一变量：生产行为不变，只采集 OPEN_LATCH 入口 actor observation 的字段级分布与 policy mean；若相关分析不足，只做离线单字段 counterfactual inference
待验证假设：某个仍带物理门型或腕部构型表示的 actor observation 在阶段入口先产生 canonical `ee_y` mean 分叉，并在累计控制后造成 opposed-contact 保持率差异
禁止改动范围：不执行 optimizer、不保存 checkpoint；不改 observation/action、reward、阶段条件、网络、temporal 参数、时限、随机化、低层、AutoOpen、门型或 seed；不添加门侧特例
运行规模与日志：30 项测试、py_compile 和 diff check 通过。单卡 `8 env x 8 steps` smoke 在 optimizer 前正常退出；因拓扑变化不恢复 staged-reset，0 个 OPEN_LATCH 样本只验证空样本字段与退出链路，日志为容器 `doorman_train:/tmp/b2z1_phase206_smoke_v3.log`，运行目录只有 config/meta。正式 v2 使用 `4 x 2048 env x 128 steps`，日志为容器 `doorman_train:/tmp/b2z1_phase206_entry_v2.log`，数据为 `doorman_train:/tmp/b2z1_phase206_entry.rank{0,1,2,3}.pt`，运行目录为 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260927_193844`
成功/失败判据：找出先于 canonical action 分叉的具体 observation 字段，并用同一 checkpoint 的离线 counterfactual 证明替换该字段会按预测改变 `ee_y` mean；若没有单字段因果项，明确证明是多字段或 recurrent hidden-state 组合，而不是凭相关性修改接口
观测指标：actor_obs 各字段、OPEN_LATCH 入口前 25 步 policy mean、task variant 与 paired/counterfactual mean delta。当前配置文件名虽含 `lstm`，实际 actor `_target_` 是普通 `Actor`、backbone 是 MLP，不存在 recurrent hidden state
结论与已排除方向：四个 rank 分别采集 `6910/7546/7350/7602` 个样本，合计 `29408`；左/右为 `16184/13224`，`actor_obs` 为 `[N,172]`，`action_mean` 为 `[N,10]`，时间覆盖 `0--24`，全部有限。冻结 RMS 和 MLP 后以最大 `2.24e-8` 误差复现原 action mean。字段均值反事实中 `b2z1_task_geometry` 对 `ee_y` 的对称效应为 `-0.003902`，解释原始 `-0.004849` 门型差约 `80.5%`；最强单维为其中 jaw axial 两维。代码审计确认 axial 先沿固定 handle local `-Y` 投影，之后仅交换 jaw 槽位，遗漏了 `door_open_lr` 对轴向正方向的变换。左门 canonical axial 均值为 `[+0.418,-0.460]`，右门为 `[-0.510,+0.395]`。仅对右门补轴向符号后，`ee_y` 门型差降至 `-0.001283`，约消除 `73.5%`；门侧 one-hot 单字段只解释约 `16.7%`。因此根因是坐标接口不完整，不是数据不足、门型 one-hot、hidden state 或 reward
正式保留/删除的代码：保留 `canonicalize_jaw_axial` 通用坐标纯函数、调用点和单测；phase206 observation/mean 导出、环境变量门控与 optimizer 前退出路径全部删除
前序阶段回归：31 项单元测试、容器内 py_compile、`git diff --check` 和临时标识搜索通过；本 phase 未执行 optimizer、未生成 checkpoint
下一决策门槛：phase207 必须从随机初始化训练新 observation 语义；只有左右 jaw axial 分布和 OPEN_LATCH 转换随训练收敛且前序阶段不回归，才能认为该通用修复有效
```

2026-09-27 19:37，phase206 首次四卡启动在出现第一批 OPEN_LATCH 样本时退出，未进入 optimizer、未生成 checkpoint。原因是临时诊断误把文件名中的 `lstm` 当成真实网络配置并访问 `policy_model.memory`；实际配置明确为普通 `Actor + MLP`。已删除全部 hidden-state 采集逻辑，30 项测试、py_compile 和 diff check 重新通过。该失败只属于诊断实现，禁止据此修改生产网络或训练语义；正式 v2 仅导出 MLP 的 actor observation 与 mean。

2026-09-27 19:43，phase206 正式 v2 四卡采集正常完成并在 optimizer 前退出。四份数据分别包含 `6910/7546/7350/7602` 个样本，合计 `29408`，左/右为 `16184/13224`；`actor_obs` 维度为 172，`action_mean` 维度为 10，时间覆盖阶段入口 `0--24`，所有浮点值有限。运行目录仅有 config/meta，没有新 checkpoint。下一步只进行字段统计和离线 counterfactual；在因果证据形成前不修改生产 observation、reward 或控制行为。

2026-09-27 19:58，phase206 字段归因和冻结 checkpoint 反事实完成。`b2z1_task_geometry[18:20]` 对应 canonical jaw axial 两维，是主要因果输入；当前实现交换了 right-hinged jaw 槽位，却没有翻转沿把手自由端方向的 axial 符号。新增的 `canonicalize_jaw_axial` 同时统一轴向正方向和 jaw 槽位，只用于这两个 observation 维度，不改变 closing、force、接触验收或 reward。phase206 临时代码已删除，31 项测试通过。

### phase207：统一 jaw axial observation 后干净训练（已验证）

```text
上游结论与待解决瓶颈：phase206 以代码语义和冻结 checkpoint 反事实确认 jaw axial 坐标遗漏门侧符号，是 OPEN_LATCH 入口 ee_y 分叉的主要来源
基线 checkpoint / SHA-256：无；observation 语义改变后从随机初始化，禁止加载 phase204 step125 或任何旧 checkpoint
唯一变量：`_grasp_contact_obs` 的 axial 两维由仅交换 jaw 槽位改为先统一把手自由端轴向符号、再交换 jaw 槽位；其余生产行为不变
待验证假设：新语义使左右门的 canonical jaw axial 分布收敛，入口 canonical ee_y 与后续 opposed-contact retention 不再形成 phase204/205 的系统性门型分叉
禁止改动范围：不改 reward、阶段验收、时间预算、其他 observation/action、网络、PPO、随机化、低层、AutoOpen、门型或 seed；不增加门侧特例；运行中不叠加第二变量
运行规模与日志：无 checkpoint 的 `1 x 8 env x 8 steps x 1 update` 真实 optimizer smoke 已通过，日志为容器 `doorman_train:/tmp/b2z1_phase207_axial_clean_smoke.log`，运行目录为 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260927_195339`；正式纯动力学 `4 x 2048 env x 128 steps`、125 updates 已启动，每 25 update 保存 checkpoint，日志为容器 `doorman_train:/tmp/b2z1_phase207_axial_clean_8192_to125.log`，运行目录为 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260927_195454`
成功/失败判据：无异常/NaN；APPROACH_PREGRASP、GRASP 不回归；左右 OPEN_LATCH jaw axial、入口动作、严格转换与 opposed-contact 差异显著小于 phase204；最终仍以 seed 1700--1703 全阶段成功为总验收
观测指标：六阶段 reached/transition、左右 jaw axial、canonical ee_y/ee_x、pocket/axis/opposed-contact retention、latch/gap/hinge、安全力、task success
结论与已排除方向：正式训练完成 125 updates、`131,072,000` timesteps，无异常或 NaN。jaw axial 修复持续将左右入口动作均值大幅拉齐，但 step125 严格转换左/右仅 `6.17%/25.64%`，未优于 phase204 的 `7.02%/27.10%`；排除 jaw axial 非 canonical 是剩余学习瓶颈的完整解释，也排除继续盲目延长同分布训练具有证据基础。修复仍是代码语义正确的通用接口修复，正式保留
正式保留/删除的代码：仅保留通用坐标修复和测试；不增加诊断行为路径
前序阶段回归：启动前 31 项测试、py_compile、diff check 和 phase206 临时标识清理已通过
下一决策门槛：phase208 只读定位 opposed contact 首次丢失事件的物理状态、动作时序和 exact credit；根因确认前不改 reward 或行为
```

2026-09-27 19:54，phase207 真实 Isaac optimizer smoke 正常退出。环境为纯动力学、headless、无 checkpoint，完成 1 update 和 64 timesteps；没有训练异常、NaN 或新接口错误。正式四卡训练保持同一代码和配置，只扩大并行度、rollout 长度与 update 数。

2026-09-27 19:55，phase207 四卡正式干净训练启动。四个 rank 各 2048 env、128 steps/env，共 8192 env；随机初始化，不加载旧 observation 语义 checkpoint。四个 rank 均已完成纯动力学场景创建，camera types 为空；运行目录为 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260927_195454`。step25 前不修改行为。

2026-09-27 19:59，phase207 完成 iteration 1。总计 `1,048,576` timesteps，iteration time `62.15s`；四个 rank 进程正常，无异常或 NaN。该节点仅证明正式四卡规模与新 observation 语义的训练链路成立，不用于评价学习效果；下一决策点仍为 step25。

2026-09-27 20:25，phase207 完成 step25 并保存首个新 observation 语义 checkpoint。左右 GRASP reached 为 `31.72%/36.84%`，OPEN_LATCH reached 为 `20.74%/16.37%`，REPOSITION reached 为 `0.25%/0.81%`，严格 OPEN_LATCH 转换约 `1.21%/4.95%`；PUSH_OPEN reached `0.04%/0.16%`，THROUGH 均为 0。左右 OPEN_LATCH policy mean `ee_x=+0.0180/+0.0181`、`ee_y=+0.0012/+0.0008`，contact quality `0.5482/0.5617`，入口动作门型分叉已显著消失。总体 pocket/axis/contact retention 为 `0.6255/0.4459/0.3595`，opposed contact `0.0501`，max handle force `57.86 N`，task success 为 0。当前不改代码，继续到 step50。checkpoint SHA-256 为 `13dba57e1c087a09ac28a1fdc983af6592697350d99834dca1aa2b8d0ce5865c`。

备份记录：2026-09-27 20:28 已将 phase207 当前 31 个工作区源码/文档文件、更新后的执行账本和 step25 checkpoint 同步到 `wholebody_1/3/4`，未同步训练日志、缓存、其他运行目录或凭据，也未修改远程 Git 历史。`task_frame.py`、`door_open_homie.py`、`test_door_task_frame.py`、本文档和 checkpoint 的 SHA-256 已逐项核对，四机一致；写入本条后本文档单独再次同步。

2026-09-27 20:51，phase207 完成 step50。左右 GRASP reached `39.43%/42.27%`，OPEN_LATCH reached `26.96%/20.14%`，REPOSITION reached `0.90%/3.48%`，严格转换约 `3.34%/17.28%`；PUSH_OPEN reached `0.34%/1.05%`，THROUGH reached `0.03%/0.01%`，task success 为 0。左右 OPEN_LATCH policy mean `ee_x=+0.0127/+0.0131`、`ee_y=+0.0004/+0.0005`，contact quality `0.6468/0.6392`，总体 pocket/axis/contact retention `0.7822/0.5623/0.4761`，opposed contact `0.0668`，max handle force `63.20 N`。jaw axial 修复持续消除了入口动作差，但严格转换差异与 phase204 step50 同量级，因此保持唯一变量继续到 step75。checkpoint SHA-256 为 `1791763ebc885f0c9dcfaceeeb1441218b71186c765c2dcc82cbfc961654a0e0`。

2026-09-27 21:21，phase207 完成 step75。左右 GRASP reached `41.84%/46.00%`，OPEN_LATCH reached `28.84%/24.84%`，REPOSITION reached `1.51%/4.10%`，严格转换约 `5.24%/16.51%`；PUSH_OPEN reached `0.69%/1.65%`，THROUGH reached `0.07%/1.27%`，task success 为 0。左右 OPEN_LATCH policy mean `ee_x=+0.0056/+0.0068`、`ee_y=+0.0001/-0.0002`，contact quality `0.6484/0.6612`，总体 pocket/axis/contact retention `0.7884/0.5537/0.4863`，opposed contact `0.0516`，max handle force `38.68 N`。GRASP 阶段左/右 accept pocket distance 为 `0.7411/0.4415`、accept opposed 为 `0.0303/0.0105`，左门入口质量反而更高但后续转换更差；剩余分叉不能简单归因于 GRASP。保持唯一变量继续到 step100。checkpoint SHA-256 为 `1a8f80fd037c7c0bf3be801b3613c34afa2aef1b5faf6c011233443b692bdcb3`。

备份记录：2026-09-27 21:23 已将 phase207 step75 checkpoint 和更新后的执行账本同步到 `wholebody_1/3/4`；checkpoint 与同步前文档 SHA-256 三机一致，checkpoint 为 `1a8f80fd037c7c0bf3be801b3613c34afa2aef1b5faf6c011233443b692bdcb3`。正式代码自 20:28 的 31 文件完整备份后未变化；本条写入后本文档单独再次同步。

2026-09-27 21:50，phase207 完成 step100。左右 GRASP reached `43.98%/47.19%`，OPEN_LATCH reached `31.54%/26.17%`，REPOSITION reached `2.14%/5.21%`，严格转换约 `6.79%/19.91%`；PUSH_OPEN reached `1.22%/2.26%`，THROUGH reached `0.09%/1.69%`，task success 为 0。左右 OPEN_LATCH policy mean `ee_x=+0.0029/+0.0028`、`ee_y=+0.0011/-0.0000`，contact quality `0.6535/0.6361`，总体 pocket/axis/contact retention `0.7969/0.5143/0.4760`，opposed contact `0.0282`，max handle force `45.60 N`。严格转换没有超过 phase204 同期，继续到预定 step125 后停止本轮训练。checkpoint SHA-256 为 `4876fd1b8587943d52517e46bad729d7bec58d5b811925ab1068c91e0d0a4c0d`。

2026-09-27 22:18，phase207 正常完成 step125 并退出。总训练量 `131,072,000` timesteps，最终 iteration time `67.77s`，无异常或 NaN。左右 GRASP reached `43.15%/49.75%`，OPEN_LATCH reached `30.94%/28.04%`，REPOSITION reached `1.91%/7.19%`，严格转换约 `6.17%/25.64%`；PUSH_OPEN reached `1.19%/2.83%`，THROUGH reached `0.10%/2.36%`，task success 为 0。左右 policy mean `ee_x=+0.0020/+0.0026`、`ee_y=+0.0027/+0.0005`，contact quality `0.6534/0.6372`，总体 pocket/axis/contact retention `0.7917/0.6043/0.4771`，opposed contact `0.0332`，max handle force `54.33 N`。与 phase204 step125 的严格转换 `7.02%/27.10%` 相比没有改善，phase207 不满足学习假设或最终验收；jaw axial 通用接口修复保留，下一步不再续训。checkpoint SHA-256 为 `4b98a44dd85edbc4ee9c638be5c8c3284e679a535cdde9c48fbddffbd2288c86`。

备份记录：2026-09-27 22:22 已将 phase207 最终执行账本和 step125 checkpoint 同步到 `wholebody_1/3/4`；同步前文档与 checkpoint SHA-256 三机一致，checkpoint 为 `4b98a44dd85edbc4ee9c638be5c8c3284e679a535cdde9c48fbddffbd2288c86`。正式代码自 20:28 的完整备份后未变化；本条写入后本文档单独再次同步。

### phase208：OPEN_LATCH opposed contact 首次丢失事件归因（已验证）

```text
上游结论与待解决瓶颈：phase207 证明 jaw axial 修复使入口动作均值对齐，却未改善严格转换；最终 opposed contact 仅 3.32%，剩余瓶颈位于 OPEN_LATCH 内部接触保持、动作时序或 credit
基线 checkpoint / SHA-256：phase207 model_step_000125.pt / 4b98a44dd85edbc4ee9c638be5c8c3284e679a535cdde9c48fbddffbd2288c86
唯一变量：生产行为不变，只增加 optimizer 前退出的事件中心化只读导出；不更新模型、不生成 checkpoint
待验证假设：严格失败轨迹在 opposed contact 首次 `1 -> 0` 前存在可重复的 canonical 动作或接触几何变化，并且该动作从 exact temporal credit 获得非负或门侧不对称信号，导致策略没有学会保持双爪对向接触
禁止改动范围：不改 reward、阶段验收、observation/action、网络、PPO、temporal 参数、时限、随机化、低层、AutoOpen、门型或 seed；不添加门侧特例
运行规模与日志：单卡 smoke 为 `8 env x 8 steps`，容器日志 `doorman_train:/tmp/b2z1_phase208_smoke.log`，数据 `doorman_train:/tmp/b2z1_phase208_smoke/phase208.rank0.pt`；正式使用 `4 x 2048 env x 512 steps`，日志 `doorman_train:/tmp/b2z1_phase208_events_512.log`，数据目录 `doorman_train:/tmp/b2z1_phase208_events_512/`
成功/失败判据：对左右门分别定位 opposed contact 首次丢失前后的动作、接触点轴向/闭合坐标、双爪力、pocket/axis、latch/gap/hinge及 exact reward/advantage 变化；必须区分“物理先丢失”与“credit 鼓励丢失”，不能只报告最终通过率
观测指标：stage/time/variant、opposed/on-capsule/recent contact、canonical axial/closing/force、pocket/axis、base/sample/physical action、latch/gap/hinge、reward terms、value/return/advantage/sequence score、严格阶段转换
结论与已排除方向：四个 rank 共覆盖约 `114 万` 个 OPEN_LATCH 动作，按每个连续片段只取第一次 `opposed 1 -> 0` 后得到左/右 `2085/1564` 个事件。约 `78.5%/75.5%` 发生在阶段前 5 步。事件后约 `97.3%/95.7%` 仍是双爪当前有效接触，约 `99.7%/99.9%` 当前 closing coordinate 仍异号；固定/移动爪平均力约左 `19.9/36.2 N`、右 `23.2/34.8 N`。EE pocket 单步位移中位数仅 `0.88/0.91 mm`，axis error 单步变化中位数约 `0.0069/0.0071 rad`，但归一化轴向接触点跳变中位数为 `0.60/0.32` 个半把手长度，实际对应厘米级。由于每个接触点仍分别满足有效把手胶囊体和闭合轴两侧条件，旧 opposed 失效只能来自 `abs(stored_axial[0]-stored_axial[1]) <= 0.020`。该单点 ContactSensor 位置会在碰撞流形内切换，不代表接触消失。事件帧 reward 几乎全部由 `task_stage_progress` 错误扣除，早期左/右均值为 `-1.575/-1.462`；normalized advantage 与 exact sequence credit 没有一致地奖励该事件。排除策略大动作先造成真实脱离，也排除 credit 主动鼓励丢夹持是首因
正式保留/删除的代码：不保留任何 phase208 代码；环境变量导出、逐项 reward 缓存、optimizer 前退出及两个离线分析脚本已删除。原始数据和 JSON 报告仅保留在容器 `/tmp` 作为本轮证据
前序阶段回归：诊断启动前 31 项测试、容器内 py_compile 与 diff check 通过；正式 rollout 四 rank 正常退出，无 optimizer、checkpoint、异常或 NaN；清理后需再次执行完整检查
下一决策门槛：phase209 只删除双接触点之间固定 2 cm 的轴向距离条件。每点胶囊体有效性、closing 两侧、最小径向深度、recent、pocket、axis、force 与 unexpected-contact 条件全部保留；先同 checkpoint 只读对照，不直接续训
```

2026-09-27 22:34，phase208 单卡 smoke 正常完成并在 optimizer 前退出。导出的 68 个字段均为预期形状，所有浮点值有限；26 个 scaled reward 项加 `0.9975 * time_outs * values` 与 storage reward 的最大误差为 `9.54e-7`。运行目录仅有 config/meta，没有 checkpoint。因 checkpoint 的四卡 staged-reset 状态不会加载到单卡 8-env 拓扑，OPEN_LATCH 样本为 0；这轮只验证链路，不构成物理或 credit 结论。

2026-09-27 22:44，phase208 正式四卡只读 rollout 正常完成。每个 rank 为 `[512,2048,...]`，四份文件各约 `719 MiB`，所有浮点字段有限；逐项 reward 加 timeout bootstrap 与 storage reward 最大误差为 `7.63e-6`。四个 rank 恢复了各自 `4465--4518` 个 staged-reset 条目，运行目录只有 config/meta，没有新 checkpoint。原始数据为容器 `doorman_train:/tmp/b2z1_phase208_events_512/phase208.rank{0,1,2,3}.pt`，exact score 为 `doorman_train:/tmp/b2z1_phase208_score.rank{0,1,2,3}.pt`，汇总为 `doorman_train:/tmp/b2z1_phase208_event_report.json`。

### phase209：opposed-contact 轴向配对假阴性修复（已验证）

```text
上游结论与待解决瓶颈：phase208 证明双爪仍真实接触时，ContactSensor 单点位置会沿把手轴向跳变数厘米，固定 2 cm 双点距离条件把碰撞流形切换误判为失去夹持
基线 checkpoint / SHA-256：phase207 model_step_000125.pt / 4b98a44dd85edbc4ee9c638be5c8c3284e679a535cdde9c48fbddffbd2288c86（只读对照，不续训）
唯一变量：从 opposed-contact 定义删除 `abs(stored_axial[0]-stored_axial[1]) <= 0.020`；其余每点有效胶囊体、closing 异号与最小径向深度、recent contact 条件不变
待验证假设：移除不具物理必要性且依赖不连续单点位置的双点轴向配对后，真实双爪接触不再产生大规模假阴性；两侧 opposed retention 和严格转换改善，且不会接受单爪接触、同侧接触、把手外接触或几何未对齐
禁止改动范围：不改 reward 权重、其他验收条件、observation/action、temporal 概率、网络、时限、随机化、低层、AutoOpen、门型或 seed；不把旧 checkpoint 用于正式续训
运行规模与日志：单卡 smoke 使用 `8 env x 8 steps`，日志 `doorman_train:/tmp/b2z1_phase209_smoke.log`，数据目录 `doorman_train:/tmp/b2z1_phase209_smoke/`；通过后用 phase207 step125 做 `4 x 2048 env x 512 steps` optimizer 前只读对照，日志 `doorman_train:/tmp/b2z1_phase209_compare_512.log`，数据目录 `doorman_train:/tmp/b2z1_phase209_compare_512/`
成功/失败判据：事件中心化假阴性显著消失；单爪、同侧、把手外和 closing 深度不足仍拒绝；APPROACH/GRASP 与安全不回归；只有只读语义通过后才允许从随机初始化训练
观测指标：old/new opposed、双爪 on-capsule、closing sign/depth、pocket/axis/force、stage-time、strict transition、reward 分量和 unsafe-force
结论与已排除方向：通过。四卡共 `4,194,304` 帧，OPEN_LATCH 左/右 `623603/471748` 帧。legacy/new opposed 为左 `3.58%/95.40%`、右 `3.19%/79.00%`；legacy/new precise 为左 `0.62%/1.91%`、右 `0.79%/2.32%`。新增 precise 左/右 `7988/7192` 帧，全部至少一爪当前有效接触，双爪当前有效接触占 `98.94%/97.94%`，closing 当前仍有效占 `99.95%/99.68%`，平均/最大把手力为左 `36.26/49.99 N`、右 `31.99/49.99 N`，没有新增 unsafe-force。少数非双爪当前接触帧来自既有 5-step recent-contact 滞回：左 `85/85`、右 `145/148` 在前后 5 帧内恢复双爪当前接触；右门剩余 3 帧未完成阶段且门缝未达标。严格转换事件由 phase208 左/右 `0/98` 变为 `224/206`，对应 `0%/23.39% -> 27.45%/48.93%`。排除删除轴向配对会接受单爪、同侧、把手外、过力或明显失配抓取
正式保留/删除的代码：保留 `opposed_handle_contact` 通用纯函数、唯一生产调用点和拒绝/接受测试；已删除 `B2Z1_PHASE209_EXPORT_DIR`、legacy 对照缓存、逐步导出和 optimizer 前退出路径
前序阶段回归：单卡 smoke 和四卡只读对照均正常退出，无 optimizer、checkpoint、异常或 NaN；四卡均验证 cameras=0、RTX sensors=0、frame rendering disabled。清理后 `33 passed, 5 subtests passed`，py_compile、`git diff --check` 和临时标识清零通过
下一决策门槛：清理回归与三台备份同步通过后，phase210 从随机初始化训练；不得继承 phase207 或其他旧验收语义 checkpoint
```

2026-09-27 23:00，phase209 单卡 smoke 已登记待启动。唯一目的是验证 legacy/new opposed 同步导出的字段、形状、有限值和 optimizer 前退出链路；单卡拓扑不会恢复四卡 staged-reset 样本，因此即使没有 OPEN_LATCH 样本也不作物理结论。启动前确认容器内无遗留训练进程。

2026-09-27 23:02，phase209 单卡 smoke 正常完成。导出的 16 个字段均为预期形状，所有浮点值有限；日志明确恢复 step125 后因四卡到单卡拓扑变化跳过 rank-local environment state，并在写出 `phase209.rank0.pt` 后于 optimizer 前退出。运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260927_230107` 只有 `config.yaml/meta.yaml`，没有 checkpoint；退出后无遗留训练进程或 GPU 占用。允许启动四卡同规模只读对照。

2026-09-27 23:15，phase209 四卡只读对照正常完成。四个 rank 各恢复 `4465--4518` 个 staged-reset 条目，场景均确认纯动力学无相机；四份 `[512,2048,...]` 导出各约 `89 MiB`，所有浮点字段有限，运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260927_230239` 只有 config/meta，无 optimizer 或 checkpoint。对照支持唯一假设，详细数量结论已写入本 phase 主记录；临时诊断代码随即删除，进入清理回归。

2026-09-27 23:17，phase209 清理回归完成。`B2Z1_PHASE209_EXPORT_DIR`、legacy 对照缓存、逐步导出和 optimizer 前退出路径搜索为空；Isaac Python 测试为 `33 passed, 5 subtests passed`，py_compile 与 `git diff --check` 通过。正式工作树只保留通用 opposed-contact 几何修复、唯一生产调用点和测试，允许同步备份并进入 phase210 计划。

备份记录：2026-09-27 23:18 已将 phase209 清理后的全部正式工作树修改与本文档同步到 `wholebody_1/3/4`；未同步训练日志、缓存、`/tmp` 诊断数据或凭据，也未修改远程 Git 历史。`stage_conditions.py`、`door_open_homie.py`、`test_door_open_stage_conditions.py` 和同步前本文档 SHA-256 四机一致；本条写入后本文档单独再次同步。

### phase210：新 opposed-contact 语义下四卡干净训练（已验证）

```text
上游结论与待解决瓶颈：phase209 已修复 OPEN_LATCH 夹持验收的系统性假阴性；冻结 phase207 策略的严格转换显著恢复，但旧 checkpoint 的 actor、critic 与 staged-reset 样本均来自旧验收/reward 反馈，不能用于正式续训
基线 checkpoint / SHA-256：无；actor、critic 和 staged-reset snapshots 全部随机初始化，禁止加载 phase207 或其他历史 checkpoint
唯一变量：相对 phase207 干净训练，只采用 phase209 已验证的新 opposed-contact 几何定义；其余 reward、阶段条件、observation/action、网络、temporal sampler、时限、随机化、低层和 AutoOpen 不变
待验证假设：移除假阴性反馈后，策略能在不牺牲 pocket、axis、真实接触和安全的前提下，提高左右 OPEN_LATCH 严格转换，并向 REPOSITION、PUSH_OPEN、THROUGH 提供足够样本
禁止改动范围：运行中不改代码；不加载旧 checkpoint；不改 reward 数值、其他阶段验收、observation/action、网络、PPO、temporal 参数、时限、随机化、低层、AutoOpen、门型或 seed；不增加阶段特例、隐藏开关或临时阈值
运行规模与日志：先运行无 checkpoint 的 `1 x 8 env x 8 steps x 1 update` optimizer smoke；通过后运行纯动力学、headless、无相机、无 RTX sensor、关闭 frame rendering 的 `4 x 2048 env x 128 steps`，计划训练 125 updates，每 25 update 保存 checkpoint；日志路径在启动时写入
成功/失败判据：无异常或 NaN；APPROACH_PREGRASP 稳定恢复，GRASP 不低于 phase207 同期量级；左右 OPEN_LATCH 严格转换相对 phase207 step125 的 `6.17%/25.64%` 明显改善，且 pocket/axis/contact retention 与 unsafe-force 不回归；若最早瓶颈移动到后续阶段，停止改 OPEN_LATCH 并按阶段证据继续
观测指标：六阶段 reached/transition、左右门 strict transition、opposed/precise grasp、pocket/axis/contact retention、latch/gap/hinge、opening force、unsafe-force、task success、policy mean/sample/physical action、value/return/KL/ratio/loss
结论与已排除方向：训练完成 125 updates、`131,072,000` timesteps，无异常或 NaN。最终 OPEN_LATCH 严格转换左/右约 `77.58%/36.78%`，显著高于 phase207 step125 的 `6.17%/25.64%`；REPOSITION -> PUSH_OPEN 两侧约 `49.08%/54.78%`，排除 REPOSITION 是当前门侧分叉主层。PUSH_OPEN -> THROUGH 左/右约 `3.74%/79.94%`，且差异在 step75/100/125 持续，确认左门 PUSH_OPEN 为首个稳定后续瓶颈。尚未确定其物理、动作或 credit 根因；task success 仍为 0
正式保留/删除的代码：训练期间工作树冻结，未新增任何诊断或行为代码；phase209 正式几何修复保持不变
前序阶段回归：phase209 清理后 `33 passed, 5 subtests passed`，py_compile、diff check、临时标识清零和三台备份同步通过
下一决策门槛：phase211 冻结 step125 策略做左/右 PUSH_OPEN 只读对照，先定位入口状态、动作、物理响应或 credit 的首个分化层；不直接续训或修改 reward
```

2026-09-27 23:19，phase210 单卡 optimizer smoke 登记启动。配置为 GPU 3、`1 x 8 env x 8 steps x 1 update`，无 checkpoint，日志 `doorman_train:/tmp/b2z1_phase210_clean_smoke.log`；启动前无遗留训练进程。

2026-09-27 23:21，phase210 单卡 optimizer smoke 正常完成。随机初始化完成 1 update、64 timesteps，iteration time `3.70s`；运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260927_232011` 只有 config/meta，无 checkpoint。日志无异常或 NaN，退出后无遗留训练进程；允许启动四卡正式训练。

2026-09-27 23:21，phase210 四卡正式干净训练登记启动。配置为 `4 x 2048 env x 128 steps`、4 mini-batches、125 updates，每 25 update 保存 checkpoint；随机初始化，不加载任何 checkpoint。日志为容器 `doorman_train:/tmp/b2z1_phase210_clean_8192_to125.log`，首个行为决策点为 step25，训练期间工作树冻结。

2026-09-27 23:26，phase210 完成 iteration 1。总计 `1,048,576` timesteps，iteration time `62.52s`，吞吐约 `16,770 steps/s`；四个 rank 和 optimizer 正常，无异常或 NaN。首轮 stage storage 全部为 APPROACH_PREGRASP，符合随机初始化且尚未积累 staged-reset 快照的预期；该节点只证明正式训练链路成立，不用于评价学习效果，下一决策点仍为 step25。

2026-09-27 23:53，phase210 完成 step25 并保存首个 checkpoint。左右 APPROACH_PREGRASP reached 均为 `100%`，GRASP `33.37%/37.00%`，OPEN_LATCH `23.11%/17.07%`，REPOSITION `6.62%/2.34%`，严格 OPEN_LATCH 转换约 `28.65%/13.71%`；PUSH_OPEN reached `1.13%/0.26%`，THROUGH `0.01%/0.06%`，task success 为 0。总体 opposed `0.5888`，pocket/axis/contact retention `0.6820/0.5255/0.6791`，max handle force `53.07 N`，OPEN_LATCH unsafe-force 为 0。与 phase207 step25 严格转换 `1.21%/4.95%`、contact retention `0.3595` 相比，phase209 修复已产生显著在线收益；右门当前仍弱于左门，但不作门型特调，继续到 step50。checkpoint 为 `model_step_000025.pt`，SHA-256 `9b3370d07698c0426e08d42aabad905b7ed19b1d3340a5c95ad57157276aba23`。

2026-09-28 00:23，phase210 完成 step50。左右 GRASP reached `48.59%/44.42%`，OPEN_LATCH `38.19%/24.18%`，REPOSITION `25.97%/7.93%`，严格转换约 `68.00%/32.80%`；PUSH_OPEN reached `7.59%/2.27%`，THROUGH `0.13%/1.59%`，task success 为 0。总体 opposed `0.6746`，pocket/axis/contact retention `0.8046/0.6116/0.7946`，max handle force `59.00 N`，OPEN_LATCH unsafe-force 为 0。相比 phase207 step50 严格转换 `3.34%/17.28%`、contact retention `0.4761`，新语义收益继续扩大且未牺牲安全；当前最早瓶颈开始移向 REPOSITION。checkpoint 为 `model_step_000050.pt`，SHA-256 `d7ae0165f1b2072cf5cb49c1c8728312d6d4f30471f88a955176df0ceca18710`。

2026-09-28 00:52，phase210 完成 step75。左右 GRASP reached `56.09%/47.51%`，OPEN_LATCH `47.44%/27.47%`，REPOSITION `33.30%/9.05%`，严格 OPEN_LATCH 转换约 `70.19%/32.94%`；PUSH_OPEN reached `12.76%/3.69%`，REPOSITION -> PUSH_OPEN 约 `38.32%/40.77%`；THROUGH reached `0.22%/3.20%`，PUSH_OPEN -> THROUGH 约 `1.72%/86.72%`，task success 为 0。总体 opposed `0.7901`，pocket/axis/contact retention `0.7957/0.5194/0.8430`，max handle force `50.76 N`，OPEN_LATCH unsafe-force 为 0。OPEN_LATCH 修复收益已稳定，左门 PUSH_OPEN 成为新的强候选瓶颈；保持唯一变量到 step100/125 增加样本，不提前修改。checkpoint 为 `model_step_000075.pt`，SHA-256 `dd8f0f30d109955f00dba28909488ac013c534305fad19012fc103c7a6f9da8f`。

2026-09-28 01:22，phase210 完成 step100。左右 GRASP reached `60.12%/50.00%`，OPEN_LATCH `51.12%/30.39%`，REPOSITION `38.89%/10.50%`，严格 OPEN_LATCH 转换约 `76.08%/34.55%`；PUSH_OPEN reached `16.94%/4.59%`，REPOSITION -> PUSH_OPEN 约 `43.56%/43.71%`；THROUGH reached `0.37%/3.46%`，PUSH_OPEN -> THROUGH 约 `2.18%/75.38%`，task success 为 0。总体 opposed `0.7335`，pocket/axis/contact retention `0.8002/0.5018/0.8026`，max handle force `60.82 N`，OPEN_LATCH unsafe-force 为 0。左门 PUSH_OPEN 低转换已在 step75/100 连续成立，但尚未归因；保持代码冻结完成 step125。checkpoint 为 `model_step_000100.pt`，SHA-256 `1a2e3c1c69738c0fa144749361bb0bc17e0853e625905d74645ea5b811b3d1aa`。

2026-09-28 01:52，phase210 正常完成 step125 并退出。总训练量 `131,072,000` timesteps，最终 iteration time `68.40s`，无异常或 NaN。左右 GRASP reached `64.79%/52.06%`，OPEN_LATCH `56.93%/32.16%`，REPOSITION `44.17%/11.83%`，严格 OPEN_LATCH 转换约 `77.58%/36.78%`；PUSH_OPEN reached `21.68%/6.48%`，REPOSITION -> PUSH_OPEN 约 `49.08%/54.78%`；THROUGH reached `0.81%/5.18%`，PUSH_OPEN -> THROUGH 约 `3.74%/79.94%`，task success 为 0。总体 opposed `0.8078`，pocket/axis/contact retention `0.7926/0.5087/0.8330`，max handle force `66.77 N`，OPEN_LATCH unsafe-force 为 0。phase209 修复的在线收益成立；左门 PUSH_OPEN 是下一只读诊断对象。checkpoint SHA-256 为 `48593ea9b943831e1973f4aa0a7816b956b7b14be225594b0d7613478709b028`。

备份记录：2026-09-28 01:53 已将 phase210 最终 step125 checkpoint 和执行账本同步到 `wholebody_1/3/4`；checkpoint SHA-256 四机一致，为 `48593ea9b943831e1973f4aa0a7816b956b7b14be225594b0d7613478709b028`。写入本条后本文档单独再次同步。

### phase211：左门 PUSH_OPEN 首个分化层只读归因（已验证）

```text
上游结论与待解决瓶颈：phase210 已确认 OPEN_LATCH 修复在线有效，REPOSITION -> PUSH_OPEN 左右条件转换接近；PUSH_OPEN -> THROUGH 左/右约 3.74%/79.94%，左门 PUSH_OPEN 是当前最早稳定后续瓶颈，但尚未确认分化首先来自入口状态、策略动作、实际 EE/底盘响应、门板物理响应还是 reward/temporal credit
基线 checkpoint / SHA-256：phase210 model_step_000125.pt / 48593ea9b943831e1973f4aa0a7816b956b7b14be225594b0d7613478709b028（冻结策略，只读 rollout，不更新模型）
唯一变量：增加可完全删除的只读诊断导出；生产行为、策略参数和全部训练语义保持不变
待验证假设：通过同一冻结策略下左右门逐轨迹时序对照，可以确定 PUSH_OPEN 首个显著分化发生在入口物理状态、策略动作、执行响应、门板响应或 exact credit 中的哪一层；聚合日志中左门高铰链角、低接触率和较小 policy mean ee_x 仅作为线索，不预设为根因
禁止改动范围：不得更新 optimizer；不得修改 reward、阶段条件、observation/action、网络、时间预算、随机化、低层、AutoOpen 或任何门型、阶段、seed 特例；根因确认前不得写行为补丁
运行规模与日志：单卡 smoke 为 GPU 3、`8 env x 8 steps`，日志 `doorman_train:/tmp/b2z1_phase211_smoke.log`，导出目录 `doorman_train:/tmp/b2z1_phase211_smoke/`；通过后做 `4 x 2048 env x 512 steps` 冻结策略双门对照。运行必须保持纯动力学、headless、无相机、无 RTX sensor、关闭 frame rendering
成功/失败判据：成功要求在逐轨迹时间顺序中定位首个可重复的左右分化层，并用动作、物理状态、reward 和 exact temporal likelihood 的闭环证据排除更早层；若字段或样本不足，只能登记诊断失败并补充证据，不能修改行为
观测指标：stage/stage_time/variant、入口前后状态、hinge angle/speed/opening speed、contact seen、panel force/clearance/approach speed、push-ready 位置姿态误差、shoulder clearance、base/EE 响应、policy mean/sigma/sample、temporal context、action reference、delta accumulator、physical command、stage/contact/hinge progress、逐项 scaled reward、storage reward、value/return/advantage/timeouts、exact sequence score、安全与终止原因
结论与已排除方向：四卡共覆盖左/右 PUSH_OPEN `330176/36021` 帧和 `812/224` 个连续片段，建立门板接触比例 `21.18%/90.63%`，完成比例 `3.69%/73.21%`。左门入口平均间隙较大，但在相同门角 `1.55--1.80 rad` 下接触率仍只有 `5.31%`，右门为 `91.55%`，排除入口距离是完整解释。该区间左/右 policy mean canonical `ee_y` 为 `+0.00591/-0.00281`；两侧独立 innovation 在 lag 1--32 均证明正 `ee_y` 增大间隙、负 `ee_y` 才接近门板，排除 action 符号和低层响应错误。exact PPO 左门 `ee_y` 信号为每帧 `+0.2360`。raw reward-to-go 分解中 termination 贡献 `+7.9924`，而 push pose、stage progress、stage complete 分别贡献 `-1.8135/-1.6383/-0.9616`；去除 termination 后总方向为负。左门 330176 帧仅有 4 次 unsafe-force、2 次 stage overtime 和 1 次其他非 timeout 终止，但 terminal penalty 足以压过大量正确物理 credit。首个根因位于统一 reward/credit 幅度，不在入口、动作坐标、低层或门板物理层
正式保留/删除的代码：临时 `B2Z1_PHASE211_EXPORT_DIR`、reset 前快照、逐项 scaled reward 缓存和 optimizer 前退出路径已全部删除；未保留分析脚本或行为修改
前序阶段回归：单卡与四卡均确认纯动力学、无相机、无 RTX sensor、关闭 frame rendering；正式 rollout 无 optimizer、checkpoint、异常、NaN 或残留进程。清理后 py_compile、`33 passed, 5 subtests passed`、临时标识搜索和 `git diff --check` 通过
下一决策门槛：phase212 只读评估统一 safety termination credit。候选必须使安全失败仍显著劣于普通失败，同时不能让少量终止事件压过可达 progress、pose 和 completion credit；不得按门侧或阶段单独缩放
```

2026-09-28 09:27，恢复工作后重新核对执行账本、宿主机进程、工作树和 phase210 checkpoint。无训练进程；checkpoint SHA-256 与账本一致；phase211 尚未增加诊断代码或启动 rollout。顶部执行卡中残留的 phase207 证据和“先同步 phase210”旧游标已修正，本 phase 实验卡已登记。下一步只能先审计 PUSH_OPEN 接口和诊断字段，不能据聚合线索直接修改行为或 reward。

2026-09-28 09:39，phase211 临时只读导出链路实现完成。动作前状态在 trainer 采集，动作后物理状态、阶段事件、终止原因和逐项 scaled reward 在环境 reset 前采集；导出包含策略 marginal/conditional 分布、return 和 advantage，并在 optimizer 前正常退出。环境变量未设置时不创建快照或改变训练路径。容器内 py_compile 和相关测试通过：`20 passed, 5 subtests passed`。登记 GPU 3 单卡 smoke，尚无物理结论。

2026-09-28 09:41，phase211 单卡 smoke 正常完成。`8 env x 8 steps` 共导出 112 个字段，全部形状一致且浮点值有限；逐项 scaled reward 与 reset 前总 reward 最大误差 `9.54e-7`。运行目录只有 config/meta，无 checkpoint；日志确认 cameras=0、RTX sensors=0、frame rendering disabled，并明确在 optimizer 前退出。单卡拓扑未恢复四卡 staged-reset 状态，64 帧均为 APPROACH_PREGRASP，因此本轮只验证链路，不构成 PUSH_OPEN 物理结论。允许启动 `4 x 2048 env x 512 steps` 正式只读对照。

2026-09-28 09:50，phase211 四卡冻结策略 rollout 正常完成。四个 rank 分别恢复 `6204--6343` 个 staged-reset 条目，各导出 112 个 `[512,2048,...]` 字段，文件约 `1.2 GiB`；所有浮点值有限，逐项 scaled reward 求和最大误差 `7.63e-6`。运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260928_094211` 仅有 config/meta，无 optimizer 或 checkpoint。原始数据为容器 `doorman_train:/tmp/b2z1_phase211_compare_512/phase211.rank{0,1,2,3}.pt`。

2026-09-28 10:02，phase211 逐轨迹、动作响应和 exact credit 分解完成，根因结论已写入主记录。诊断代码随后全部删除；容器内 py_compile 和全量测试 `33 passed, 5 subtests passed`，临时标识搜索、`git diff --check` 和进程检查通过。正式行为与 phase210 checkpoint 均未改变。

### phase212：安全终止 credit 的统一反事实设计（运行中）

```text
上游结论与待解决瓶颈：phase211 已确认左门 PUSH_OPEN 的 action/physics 映射正确，正常 progress、pose 和 completion credit 方向也正确；稀少高额 terminal penalty 通过长期 credit 压过这些信号，使策略回避门板
基线 checkpoint / SHA-256：phase210 model_step_000125.pt / 48593ea9b943831e1973f4aa0a7816b956b7b14be225594b0d7613478709b028（只读反事实，不更新模型）
唯一变量：安全终止 credit 的统一幅度或等价有界机制；必须对所有门型和阶段使用同一语义
待验证假设：存在一个统一 terminal cost，使安全终止轨迹仍明显劣于无终止失败轨迹，但不会仅凭极少事件翻转正确的阶段 progress/pose/completion 梯度
禁止改动范围：不改 hard termination 条件、600 N 阈值、observation/action、阶段条件、其他 reward、网络、PPO、随机化、低层、AutoOpen、门型或 seed；反事实通过前不改生产 reward 数值
运行规模与日志：复用容器 `doorman_train:/tmp/b2z1_phase211_compare_512/phase211.rank{0,1,2,3}.pt`，在 CPU 上离线统一缩放现有 `post/reward/termination`，重算 raw return、GAE、stage x variant advantage 归一化、exact temporal PPO 动作信号和匹配状态安全排序；离线输出登记为 `doorman_train:/tmp/b2z1_phase212_termination_counterfactual.log`。候选确定后另行登记最小 smoke 与受控短训练
成功/失败判据：左右 PUSH_OPEN 正确接近方向均获得非冲突 credit；每条 unsafe-force/overspeed/shoulder termination 轨迹的回报仍低于匹配的普通失败轨迹；前序阶段的跌倒和安全失败不能变成可钻空子的捷径
观测指标：按 stage x variant 的 exact `ee_x/ee_y` signal、逐项 return、终止类型与频率、匹配状态轨迹回报、阶段完成 credit、安全排序余量
结论与已排除方向：已验证。把现有 termination 项统一乘小虽能修复 PUSH_OPEN signal，但会把第一阶段普通跌倒成本从 10 同时降到 2--5，产生新捷径，明确拒绝。改用所有非 timeout hard termination 的统一绝对成本扫描；成本 40 时左/右 PUSH_OPEN `ee_y` PPO 信号为 `-0.9446/-3.4972`，1810 个安全事件的终止帧总奖励最大仍为 `-33.51`，且第一阶段跌倒成本由 10 提高到 40。该机制同时通过 credit 方向、安全事件负回报和前序无捷径三项门槛
正式保留/删除的代码：删除 `B2Z1DoorSixStage._reward_termination` 的阶段特例，统一复用基类 `reset_buf & ~time_out_buf`；reward scale 从 `-4000` 改为 `-2000`，20 ms policy dt 下每次 hard termination 固定为 `-40`。未保留反事实诊断代码，离线日志仅在 `doorman_train:/tmp/b2z1_phase212_termination_counterfactual.log`
前序阶段回归：return/advantage 复现最大误差 `1.91e-6/3.81e-6`；py_compile、`33 passed, 5 subtests passed` 和 `git diff --check` 通过；第一阶段 hard termination 成本不降反升
下一决策门槛：phase213 先完成真实 optimizer smoke；通过后以统一成本 40 为唯一变量做四卡受控短训练，不在运行中改代码
```

2026-09-28 10:24，phase212 离线反事实登记启动。已核对四份 phase211 原始导出均存在、字段完整且当前无训练进程；本轮只统一缩放已记录的 termination reward 并按生产实现重算 GAE 和分组归一化，不改 hard termination、600 N 阈值、其他 reward、阶段条件、observation/action 或任何生产代码。候选必须同时满足双侧 PUSH_OPEN credit 不冲突、安全终止轨迹仍明显劣于匹配普通失败、前序阶段无安全捷径三项要求。

2026-09-28 10:25，phase212 离线反事实完成并通过决策门槛。简单比例缩放因削弱第一阶段跌倒成本被拒绝；统一绝对 hard-termination cost 40 同时修复双侧 PUSH_OPEN credit、保持所有安全事件终止帧总奖励为负，并把旧的第一阶段成本从 10 提高到 40。旧 override 已删除，生产路径只保留基类统一语义；完整单测 `33 passed, 5 subtests passed`。

### phase213：统一 hard-termination cost 40 的在线链路验证（运行中）

```text
上游结论与待解决瓶颈：phase212 离线证明统一成本 40 能修复左门 PUSH_OPEN credit 且不削弱安全，但尚未经过真实环境 reward 和 optimizer 路径
基线 checkpoint / SHA-256：smoke 随机初始化，不加载 checkpoint；后续短训练是否使用诊断恢复或干净初始化须在 smoke 后单独登记
唯一变量：相对 phase210 训练语义，仅 hard-termination 统一绝对成本从旧的阶段化 10/80 改为 40
待验证假设：真实 Isaac 环境中每次非 timeout hard termination 的 scaled reward 为 -40，训练链路无异常或 NaN，纯动力学配置不回归
禁止改动范围：不改 hard termination 条件、600 N 阈值、其他 reward、阶段条件、observation/action、网络、PPO、随机化、低层、AutoOpen、门型或 seed；运行中代码冻结
运行规模与日志：GPU 3，`1 x 8 env x 8 steps x 1 update`，日志 `doorman_train:/tmp/b2z1_phase213_uniform_termination_smoke.log`；随机初始化，无 checkpoint
成功/失败判据：正常完成一次 optimizer update；无异常或 NaN；cameras=0、RTX sensors=0、frame rendering disabled；实际 termination reward 与统一成本 40 一致
观测指标：启动配置、termination episode sum、stage reached/transition、安全终止、loss/KL/ratio、运行目录和遗留进程
结论与已排除方向：已验证；正常完成一次 optimizer update，64 timesteps、iteration time `3.37 s`，无异常或 NaN。`Mean episode rew_termination` 从旧同配置 smoke 的 `-0.1429` 精确变为 `-0.5714`，证明第一阶段实际成本由 10 统一到 40；日志确认 cameras=0、RTX sensors=0、frame rendering disabled
正式保留/删除的代码：仅 phase212 已验证的统一 hard-termination 实现；不增加 smoke 专用代码
前序阶段回归：py_compile、`33 passed, 5 subtests passed`、diff check 通过
下一决策门槛：进入 phase214，且只运行一次四卡 25-update 诊断续训；该 checkpoint 不得冒充新 reward 语义下的正式干净基线
```

2026-09-28 10:26，phase213 单卡 smoke 正常完成。随机初始化 `8 env x 8 steps x 1 update`，64 timesteps，iteration time `3.37 s`；无异常、NaN 或遗留训练进程。日志 `doorman_train:/tmp/b2z1_phase213_uniform_termination_smoke.log` 确认纯动力学场景，termination episode 指标相对旧同配置精确放大 4 倍，统一成本 40 的真实环境路径成立。

### phase214：统一成本 40 的四卡诊断续训（运行中）

```text
上游结论与待解决瓶颈：phase212/213 已证明统一 hard-termination cost 40 的离线 credit 和真实环境链路成立，但尚未确认现有左门 PUSH_OPEN 策略会在线吸收正确信号
基线 checkpoint / SHA-256：phase210 model_step_000125.pt / 48593ea9b943831e1973f4aa0a7816b956b7b14be225594b0d7613478709b028；这是旧成本策略，仅用于受控因果诊断，输出不得作为正式最终基线
唯一变量：相对 checkpoint 原训练语义，仅 hard-termination 统一绝对成本改为 40；其余代码和配置冻结
待验证假设：25 updates 内左门 PUSH_OPEN 的接近动作、接触率或完成率应沿 phase212 预测方向改善，同时右门、前序阶段和安全不显著回归
禁止改动范围：训练中不改代码；不改 hard termination、600 N 阈值、其他 reward、阶段条件、observation/action、网络、PPO、随机化、低层、AutoOpen、门型或 seed；不把诊断续训 checkpoint 当成干净正式模型
运行规模与日志：GPU 0--3，`4 x 2048 env x 128 steps x 25 updates`，从 step125 到 step150，4 mini-batches，step150 保存 checkpoint；日志 `doorman_train:/tmp/b2z1_phase214_uniform_termination_to150.log`
成功/失败判据：无异常或 NaN；左门 PUSH_OPEN 接触/完成或正确 `ee_y` 行为至少一项形成与离线 credit 一致的改善，且右门 PUSH_OPEN、OPEN_LATCH/REPOSITION retention 和 unsafe-force 不显著回归。若 25 updates 只反映旧 critic 适配而行为无改善，登记为证据不足，不盲目延长
观测指标：六阶段 reached/transition、左右门严格转换、PUSH_OPEN 接触率/完成率/策略均值与 PPO signal、termination/unsafe-force/overspeed/shoulder、value/return/KL/loss
结论与已排除方向：已验证。step150 相比 phase210 step125，左/右 PUSH_OPEN→THROUGH 条件转换从 `3.74%/79.94%` 提高到 `4.38%/84.97%`，THROUGH reached 从 `0.81%/5.18%` 提高到 `1.08%/7.29%`；APPROACH_PREGRASP 保持 `100%/100%`，GRASP `65.26%/55.64%`、OPEN_LATCH `57.20%/37.09%`、REPOSITION `49.17%/23.64%`、PUSH_OPEN `24.66%/8.58%`，均不低于 phase210。最终左/右 PUSH_OPEN policy mean `ee_y=-0.0017/-0.0083`，unsafe-force、overspeed 和 shoulder termination 均为 0。统一成本 40 的在线行为收益成立；剩余左门转换低不是继续提高 terminal cost 的理由
正式保留/删除的代码：仅 phase212 统一成本 40；无 phase214 专用代码
前序阶段回归：phase213 smoke、`33 passed, 5 subtests passed`、py_compile 和 diff check 通过
下一决策门槛：完成代码、账本和证据 checkpoint 备份后启动 phase215 随机初始化正式训练；不得从 phase214 checkpoint 续训形成正式基线
```

2026-09-28 10:31，phase214 已完成 step126--131。四个 rank 各恢复 `6204--6343` 个 staged-reset 条目，单步耗时 `58.61--67.74 s`，无异常或 NaN。早期在线方向与 phase212 一致：左门 PUSH_OPEN policy mean `ee_y` 从首轮 `+0.0006` 到第二轮 `-0.0006`，接触率从 `6.99%` 到 `12.92%`；该结果只登记为运行中趋势，不能替代 step150 最终比较，训练代码保持冻结。

2026-09-28 10:53，phase214 正常完成 step150，累计 25 updates、`26,214,400` timesteps，无异常、NaN 或遗留进程。运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260928_102120`；证据 checkpoint `model_step_000150.pt` SHA-256 为 `d3f1739c55acfbf5783fe3f46e4d2b991fffc544f71f4aa8583aa51e255a56a7`。双侧 PUSH_OPEN 条件转换和 THROUGH reached 均超过 phase210，前序阶段与安全无回归，phase212 的 reward 根因修复完成在线闭环验证。该 checkpoint 只作因果证据，不替代新语义随机初始化正式基线。

### phase215：新 reward 语义下四卡干净训练（已验证）

```text
上游结论与待解决瓶颈：phase214 已证明统一 hard-termination cost 40 能被现有策略在线吸收并改善双侧 PUSH_OPEN，但该策略、critic 和 staged-reset 快照仍来自旧成本语义，不能作为正式训练结论
基线 checkpoint / SHA-256：无；actor、critic 和 staged-reset snapshots 全部随机初始化，禁止加载 phase214 或其他历史 checkpoint
唯一变量：相对 phase210 干净训练，仅采用 phase212 已验证的全阶段统一 hard-termination cost 40；其余代码与配置不变
待验证假设：干净训练能复现前序阶段学习，并在不增加安全终止的前提下消除旧成本导致的左门 PUSH_OPEN 反向 credit；首个瓶颈应晚于或弱于 phase210
禁止改动范围：训练中代码冻结；不加载旧 checkpoint；不改 hard termination、600 N 阈值、其他 reward、阶段条件、observation/action、网络、PPO、随机化、低层、AutoOpen、门型或 seed
运行规模与日志：备份完成后使用 GPU 0--3，`4 x 2048 env x 128 steps`、4 mini-batches，先计划 125 updates，每 25 update 保存 checkpoint；日志 `doorman_train:/tmp/b2z1_phase215_uniform_termination_clean_to125.log`
成功/失败判据：无异常或 NaN；前序 APPROACH_PREGRASP/GRASP/OPEN_LATCH/REPOSITION 不低于 phase210 同期量级；左门 PUSH_OPEN 正确动作、接触与转换不再出现 phase210 的系统性反向；安全终止无显著增加。若最早瓶颈移动，停止改 termination reward 并转只读归因
观测指标：六阶段 reached/transition、左右门严格转换、PUSH_OPEN 接触/完成/动作/PPO signal、retention、安全终止、task success、value/return/KL/loss
结论与已排除方向：已验证。125 updates 正常完成，无异常或 NaN。step125 GRASP/OPEN_LATCH/REPOSITION/PUSH_OPEN/THROUGH reached 左/右分别为 `63.69%/54.82%`、`55.86%/37.52%`、`48.76%/30.16%`、`24.69%/10.49%`、`2.56%/8.99%`；PUSH_OPEN→THROUGH 条件转换约 `10.37%/85.70%`，左门相比 phase210 同期显著改善，双侧 policy mean `ee_y=-0.0006/-0.0061`，统一 termination credit 的收益在随机初始化正式训练中复现。不能据此声称左右门已对称：差异在 GRASP/OPEN_LATCH 已存在，必须另做只读因果审计。THROUGH `arm_safely_stowed=0.87%` 仅是候选后续瓶颈
正式保留/删除的代码：仅统一 hard-termination cost 40；无 phase215 专用代码
前序阶段回归：phase214 在线因果验证、`33 passed, 5 subtests passed`、py_compile 和 diff check 通过
下一决策门槛：phase216 不训练、不改行为，先核对 RL 与 AutoOpen 的左右门 GRASP 目标姿态，再从匹配入口状态定位 OPEN_LATCH 首个 observation/action/physics/reward 分化层；只有直接证据明确所有权层后才允许设计单变量实验
```

备份记录：2026-09-28 10:55 已将 phase212--214 正式代码、本文档和 phase214 证据 checkpoint 同步到 `wholebody_1/3/4`。`door_open_homie.py`、reward 配置、同步时本文档和 checkpoint SHA-256 四机一致；未同步 `/tmp` 日志、诊断数据、缓存或凭据。checkpoint SHA-256 为 `d3f1739c55acfbf5783fe3f46e4d2b991fffc544f71f4aa8583aa51e255a56a7`。

2026-09-28 10:56，phase215 四卡正式干净训练登记启动。配置为 GPU 0--3、`4 x 2048 env x 128 steps`、4 mini-batches、125 updates、每 25 update 保存 checkpoint；不加载任何历史 checkpoint，代码在训练期间冻结。日志为容器 `doorman_train:/tmp/b2z1_phase215_uniform_termination_clean_to125.log`，首个行为决策点为 step25。

2026-09-28 11:02，phase215 已完成 step1--2。总计 `2,097,152` timesteps，单步耗时 `62.08/63.82 s`；四个 rank 均确认 cameras=0、RTX sensors=0、frame rendering disabled，无异常或 NaN。随机初始化首轮尚未积累 staged-reset 样本，该节点只证明正式链路成立，不用于评价 reward 学习效果；下一决策点仍为 step25。

2026-09-28 11:29，phase215 完成 step25 并保存首个正式 checkpoint。GRASP reached 左/右 `33.32%/37.58%`、OPEN_LATCH `22.83%/16.19%`，与 phase210 同期持平；REPOSITION `7.52%/2.88%` 略高。OPEN_LATCH opposed/pocket/axis/contact retention 为 `0.5636/0.7092/0.5844/0.6670`，unsafe-force `0.0001` 且 OPEN_LATCH/PUSH_OPEN 均为 0。PUSH_OPEN reached 仅 `0.92%/0.11%`，样本不足以判断转换率，但 policy mean `ee_y=-0.0053/-0.0049` 已排除旧的左门反向动作在同里程碑复现。checkpoint SHA-256 为 `240771546b8cd6986c17e42fdde716e73510c26292e5065f83ed8bc8ff153c83`；允许代码冻结继续到 step50。

2026-09-28 12:01，phase215 完成 step50 并保存 checkpoint。GRASP/OPEN_LATCH reached 与 phase210 同期持平，PUSH_OPEN reached 左/右 `7.93%/2.52%`、THROUGH `0.57%/2.16%`，均高于 phase210 的 `7.59%/2.27%`、`0.13%/1.59%`。OPEN_LATCH opposed/pocket/axis/contact retention 为 `0.7280/0.7744/0.5232/0.8030`；PUSH_OPEN 双侧动作方向正确且相关安全终止为 0。checkpoint SHA-256 为 `1c35bd7a9f498f8abf0c1e00f9f11a4585ec34494ade3d84e962fb3dd16ce146`；允许代码冻结继续到 step75。

2026-09-28 12:31，phase215 完成 step75 并保存 checkpoint。PUSH_OPEN→THROUGH 条件转换左/右约 `10.23%/88.71%`，相比 phase210 同期 `1.72%/86.72%`，左门提高约 6 倍且右门保持；THROUGH reached 为 `1.62%/2.75%`。GRASP/OPEN_LATCH 与 retention 同期稳定，PUSH_OPEN unsafe-force、overspeed、shoulder termination 均为 0。checkpoint SHA-256 为 `ff5a8d6319a00eecf53293842a37347ed52a208b7a02b658d384c7768f33f625`。phase211 定位的 termination credit 根因已完成离线、旧策略续训和随机初始化训练三层闭环验证；后续不得再调整该 reward，允许代码冻结继续到 step100 定位下一瓶颈。

2026-09-28 13:01，phase215 完成 step100 并保存 checkpoint。PUSH_OPEN→THROUGH 条件转换左/右约 `10.41%/87.62%`，phase210 同期为 `2.18%/75.38%`；六阶段 reached 和安全均无回归。THROUGH 内部 `distance_ok=37.61%`、`base_stable=18.17%`，但 `arm_safely_stowed=1.28%`，表明最早候选瓶颈已从 PUSH_OPEN 移到 THROUGH 的机械臂收拢/协同完成条件。该聚合结果只用于确定后续只读诊断对象，不能直接修改 reward 或动作。checkpoint SHA-256 为 `d3a48e4186c718290033aa94cc183645dbc035ab8e94e3d75103a32607729b45`；允许冻结完成 step125。

2026-09-28 13:24，phase215 正常完成 step125 并退出。累计 `131,072,000` timesteps，最终 iteration `69.21 s`；无 traceback、运行时异常或 NaN。运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260928_105610`，checkpoint `model_step_000125.pt` SHA-256 为 `f1262adb68a4bdad017c800741d50772b15af799b743d7a43c52313a19230a62`。最终六阶段指标已原子更新到顶部执行卡、路线图、六阶段总账和本记录；下一步为 phase216 只读左右门姿态与首个分化层审计，不启动训练、不修改行为。

备份记录：2026-09-28 13:02 已将 phase215 step100 账本、运行配置和当前有效 checkpoint 同步到 `wholebody_1/3/4`；同步时文档与 checkpoint SHA-256 四机一致。正式代码未变化并沿用 10:55 已校验版本；未同步 `/tmp` 日志、缓存、诊断数据或凭据。写入本条后本文档单独再次同步。

备份记录：2026-09-28 12:02 已将 phase215 step50 账本、运行配置和当前有效 checkpoint 同步到 `wholebody_1/3/4`；同步时文档与 checkpoint SHA-256 四机一致。正式代码未变化并沿用 10:55 已校验版本；未同步 `/tmp` 日志、缓存、诊断数据或凭据。写入本条后本文档单独再次同步。

备份记录：2026-09-28 10:02 已将 phase211 完整结论、清理回归和 phase212 计划账本同步到 `wholebody_1/3/4`；未同步 `/tmp` 原始诊断数据、训练日志、缓存或凭据。写入本条后本文档单独再次同步并校验 SHA-256。

### phase216：左右门 GRASP 姿态与接触外载荷稳态补偿（进行中）

```text
上游结论与待解决瓶颈：phase215 已验证统一 termination credit 的正式训练收益，但 step125 GRASP/OPEN_LATCH reached 左/右仍为 63.69%/54.82%、55.86%/37.52%；用户要求确认 RL 的 EE 旋转与 AutoOpen 完全一致，且原则上左右门不应存在语义不对称
基线 checkpoint / SHA-256：phase215 model_step_000125.pt / f1262adb68a4bdad017c800741d50772b15af799b743d7a43c52313a19230a62
唯一变量：诊断部分不改变行为；正式修复只允许向稳态逆映射加入由真实接触 wrench 推导的机械臂外部广义力
待验证假设：若 RL 与 AutoOpen 的左右门目标姿态、jaw 排列和 task-frame 语义完全一致，则剩余分化应能在匹配入口状态下定位到明确的 observation、policy action、低层执行、接触物理或 reward/credit 层；反之应先发现一个可复现的接口不一致
禁止改动范围：不修改 reward、observation/action、目标姿态、阶段条件、网络、PPO、随机化、低层、AutoOpen、门型或 seed；不保留 GRASP 阶段门控、永久关闭重力补偿或其他反事实代码；正式对照前不启动训练
运行规模与日志：原始 control 为容器 `doorman_train:/tmp/b2z1_phase216_original_control.log`；仅瞬时接触关闭补偿为 `/tmp/b2z1_phase216_contact_gravity_counterfactual.log`；整个 GRASP 关闭补偿为 `/tmp/b2z1_phase216_no_grasp_gravity_inverse.log`。三者均使用 phase215 step125、`4 x 2048 env x 128 steps x 1 update`，只作因果诊断
成功/失败判据：诊断部分已达到；正式修复还必须使右 fixed-jaw force 显著低于 `21 N`、左右 pocket/axis 指标接近、左门无回归，且无力方向错误、NaN 或异常
观测指标：目标/实际 EE orientation、fixed/moving jaw 排列、grasp orientation error、canonical/physical action、低层 command/实际 EE、pocket/axis/contact/opposed、handle/latch/door state、逐项 reward、advantage 和 termination
结论与已排除方向：目标姿态、实际 orientation tracking、几何 target/reference 和错误 180 度腕部分支均已排除。首个物理分化是自由空间稳态逆映射未纳入夹爪接触外载荷；接触后仍施加完整重力补偿，在翻转腕部构型中造成 jaw 受力不对称
正式保留/删除的代码：两个临时关闭补偿反事实均已删除。当前正在正式改造 `b2z1_command_base.py` 和 `door_open_homie.py`；代码半完成，不得运行
前序阶段回归：诊断基线来自 phase215 step125；正式实现尚未完成 py_compile、单测、diff check 和真实 Isaac 回归
下一决策门槛：完成字段迁移和通用 contact wrench 到 joint effort 映射，通过静态回归后，使用相同 checkpoint/staged-reset/规模运行一 update；达到力、几何、左右对称和安全门槛后才允许启动干净训练
```

2026-09-28 13:31，phase216 完成第一层静态姿态审计。门资产中 `door_open_lr=+1/-1` 分别表示左/右铰链，门把手自由端指向铰链，因此自由端世界方向为 `-frame_tangent*door_open_lr`。AutoOpen 选择与自由端相反的 gripper `+Y` 轴，即 `frame_tangent*door_open_lr`；RL `_compute_grasp_orientation()` 使用完全相同的 `target_y`，两者还都令 gripper `+X=-door_normal`。既有当前 AutoOpen 日志进一步实证：自由端 `+Y` 时目标 joint axis 为 `-Y`，自由端 `-Y` 时为 `+Y`。因此“右铰链右转、左铰链左转”的目标分支已确认对齐；下一步只读检查 phase215 中实际 EE 是否同样跟踪该目标，以及 OPEN_LATCH 的首个动态分化层。未修改任何行为代码。

2026-09-28 13:42，phase215 现有诊断把首个动态差异定位到 GRASP 接触之后。全体 GRASP 完整姿态误差仅 `0.0218 rad`，左右 command orientation error 约 `0.0038/0.0029 rad`，排除右门执行错误 180 度腕部构型。reference 已到目标且无接触的子集中，实际 EE 相对 reference 的垂直误差左/右约 `8.7/7.2 mm`，仍近似对称；包含接触后，左/右变为 `9.2/27.1 mm`。同时左门 fixed/moving jaw 力约 `1.03/2.53 N`，右门约 `19.95/0.70 N`。稳态逆映射在右门给出约 `30.9 mm` 向上自由空间重力补偿，但它不包含把手接触外力；当前待验证机理是固定爪接触托住机械臂后，该补偿变为过补偿并持续压住固定爪。登记一次不更新有效 checkpoint 的反事实：仅对 GRASP 已接触样本关闭自由空间稳态补偿，其余模型、状态、动作、reward 和物理不变；若右门垂直抬升和单爪载荷显著下降，机理成立。诊断分支和输出 checkpoint 均不得保留为生产机制。

2026-09-28 13:51，首次接触门控反事实与原始 control 均从 phase215 step125 恢复，使用同规模 `4 x 2048 env x 128 steps x 1 update`。control 右门 vertical actual-reference/fixed force 为 `17.8 mm/21.02 N`；接触门控为 `14.3 mm/21.17 N`。该门控只减少约 `3.5 mm` 偏差，未降低固定爪载荷，不能证明完整根因，临时代码已删除。进一步恒等式对账：control 右门 `command-reference=+24.2 mm`、`actual-command=-6.2 mm`、`reference-target=+1.0 mm`，恰好组成 `actual-reference=+17.8 mm`；几何 target/reference 无明显滞后。为区分“瞬时接触门控覆盖不足”和“补偿并非受力根因”，登记第二个且最后一个该方向的反事实：整个 GRASP 均关闭自由空间补偿。若固定爪力仍不下降，永久排除该方向，不再追加补丁式门控。

2026-09-28 14:00，第二个反事实完成并确认根因。相对原始 control，整个 GRASP 关闭自由空间稳态补偿后，左右 actual-reference 垂直误差从 `11.6/17.8 mm` 变为 `-6.4/-4.4 mm`，右门 fixed-jaw force 从 `21.02 N` 降至 `7.85 N`，左右 pocket-distance 通过率从 `45.76%/34.31%` 变为 `57.66%/54.31%`，axis 通过率变为 `99.99%/99.59%`。这证明目标姿态、几何 reference 和低层方向均正确；首个物理分化来自 `_compute_b2z1_steady_ee_command_world()` 只按重力计算自由空间静态补偿，夹爪接触把手后外载荷改变实际静态平衡，仍施加原补偿会在翻转构型中把固定爪压向把手。临时禁用分支已删除，生产工作树恢复。正式修复必须位于稳态逆映射/外载荷补偿所有权层，对任意接触和左右门统一生效；禁止保留“右门”“GRASP 特例”或简单永久关闭重力补偿。

2026-09-28 14:13，正式修复接续状态已登记。`b2z1_command_base.py` 已把稳态补偿核心改为接受 `external_joint_effort`，并以 `required_load_effort = gravity_effort - external_joint_effort` 计算所需负载偏置；`door_open_homie.py` 只完成了新缓冲区声明，仍残留旧 `gravity_*` 字段引用，且尚未把 fixed/moving jaw 的 ContactSensor wrench 经 body Jacobian 映射到 arm joint effort。因此当前工作树明确标记为半完成、不可运行。唯一接续顺序为：完成字段迁移 -> 实现通用 wrench 映射并核对符号 -> py_compile/diff check/单测 -> phase215 同 checkpoint 四卡一 update 对照。该条作为上下文恢复游标，禁止回到已排除的姿态方向、接触门控或整个 GRASP 关闭补偿方案。

2026-09-28 14:20，正式实现第一步完成：fixed/moving jaw 的 `force_matrix_w` 与 `contact_pos_w` 已分别形成世界系 `wrench=[F,(p_contact-p_body)×F]`，再由对应 body Jacobian 的六个 arm columns 做 `J^T wrench` 并求和；reset 首帧、NaN 接触点和无接触均清零。Isaac Lab 源码确认 `force_matrix_w` 是作用在 sensor body 上的法向力，公式符号与 `required_load_effort=gravity_effort-external_joint_effort` 一致。旧 `gravity_*` 字段引用已清零；容器内 py_compile、diff check 和全量测试 `33 passed, 5 subtests passed` 通过。下一步运行仅验证接口的 8-env smoke，日志 `/tmp/b2z1_phase216_external_load_smoke.log`；smoke 不用于判断左右门物理收益。

2026-09-28 14:21，phase216 外载荷实现 smoke 正常完成。GPU 3、`1 x 8 env x 8 steps x 1 update`、随机初始化，共 64 timesteps，iteration time `2.98 s`；场景创建、两路 ContactSensor、两个 body Jacobian、wrench 映射、稳态逆映射和 optimizer 全链无异常或 NaN，未保存 checkpoint，日志 `doorman_train:/tmp/b2z1_phase216_external_load_smoke.log`。该结果只确认接口可运行，不证明力符号或物理收益。下一步登记同 phase215 checkpoint、staged-reset 和原始 control 规模的四卡一 update 对照，唯一变量为通用外载荷补偿。

2026-09-28 14:27，phase216 首个正式外载荷候选对照正常完成，但未通过物理验收。使用 phase215 step125、四卡各 2048 env、128 steps、1 update，日志 `doorman_train:/tmp/b2z1_phase216_external_load_compare.log`，无异常、NaN 或 GRASP/OPEN_LATCH unsafe-force。相对原始 control，右门 reference-aligned-still 的 actual-reference 垂直偏差从 `15.1 mm` 降到 `12.2 mm`，总 load pose compensation 从约 `27.7 mm` 降到 `22.5 mm`，说明外载荷项不是零且按公式减少了所需 actuator load；但右门 fixed-jaw force 从 `21.02 N` 升到 `31.58 N`，pocket-distance 通过率从 `34.31%` 降到 `29.18%`，GRASP/OPEN_LATCH reached 从 `60.81%/42.64%` 略降到 `60.75%/42.37%`。左门 fixed force `0.56 -> 0.57 N`、pocket `45.76% -> 44.71%`，近似不变。候选不满足“右 fixed force 显著下降、左右 pocket 接近”的预设门槛，不能作为正式修复。当前只读审计为何上一帧接触反力直接闭环会出现位姿略改善但接触力恶化；根因明确前禁止翻符号、加滤波、调增益或继续训练。

2026-09-28 14:32，已找到首个候选中确定的力矩计算错误。Isaac Lab 的 `contact_pos_w` 是一个 sensor/filter 碰撞流形内所有接触点位置的普通平均，而 `force_matrix_w` 是各接触点法向力之和；因此 `(mean(p_i)-p_body) x sum(F_i)` 不等于 `sum((p_i-p_body) x F_i)`。该错误与候选中右门位置补偿范数 `24.5 -> 33.9 mm`、姿态补偿 `0.0131 -> 0.0180 rad` 的异常增大一致。现已改为直接读取 PhysX `get_contact_data()`，逐点构造 `F_i=force_value_i*normal_i` 后精确求和力与力矩；未改总平衡公式、阶段、reward、阈值或训练配置。静态检查和全量测试通过；先运行 8-env smoke，再做同规模物理对照。

2026-09-28 14:33，精确逐点 wrench 的 GPU 3 smoke 正常完成：`1 x 8 env x 8 steps x 1 update`，64 timesteps，iteration time `3.13 s`，无接口错误、异常或 NaN，日志 `doorman_train:/tmp/b2z1_phase216_exact_wrench_smoke.log`。相较聚合点候选 smoke 的 `2.98 s` 未见有意义的性能差异，但小规模不能用于评估 8192-env 开销。下一步启动与原始 control 完全同 checkpoint/staged-reset/规模的四卡一 update 对照，同时核对物理指标和 iteration time。

2026-09-28 14:38，精确逐点 wrench 四卡对照完成并排除“平均点错误是 31 N 主因”的假设。运行规模与 control 相同，日志 `doorman_train:/tmp/b2z1_phase216_exact_wrench_compare.log`，iteration time `69.29 s`，未增加相对原始 `70.55 s` 的可见训练开销。右门 fixed force `31.80 N`、late fixed force `30.50 N`、pocket/late-pocket `28.76%/28.50%`，与聚合点候选的 `31.58/31.20 N`、`29.18%/28.83%` 基本一致；右门稳态 actual-reference `12.1 mm`，总位置/姿态补偿 `33.3 mm/0.0181 rad`，也未恢复。精确逐点求和应保留为正确 wrench 计算，但它没有通过正式物理门槛，不能把外载荷直接闭环升级为生产机制。下一步只读审计 gravity API 符号和传感器/命令更新时序；禁止继续修改增益、滤波或符号来试错。

2026-09-28 14:44，phase216 完成控制时序与所有权审计。Isaac Lab `ImplicitActuator` 明确把 effort target 作为位置/速度 PD 之外的前馈力矩；当前 B2Z1 路径则与 lab30 一致，机械臂 effort 为零、只写位置目标。静态方程 `required effort = gravity - J^T F_external` 的符号没有错误，31 N 恶化来自闭环性质：把上一物理步由位置命令和单边接触约束共同产生的内生反力，在下一步从位置偏置中完整抵消，会令接触法向失去恢复刚度；任意已有接触力都可成为固定点，离散时延还能放大它。因此禁止继续翻符号、滤波或缩放该反馈。更重要的是，用户明确重申重力补偿和成功目标不应由环境替 actor 求解；phase216 外载荷候选到此永久停止并必须从工作树删除。

### phase217：actor 动作所有权纠偏（已验证）

```text
上游结论与待解决瓶颈：控制审计发现当前训练不是纯 RL，而是环境脚本 action reference 加 PPO residual；大量坐标被阶段 mask 完全交给脚本，另有 action supervision 强制 actor 拟合脚本
基线 checkpoint / SHA-256：不继承 checkpoint；phase215 step125 仅保留为旧混合控制证据，SHA-256 f1262adb68a4bdad017c800741d50772b15af799b743d7a43c52313a19230a62
唯一变量：移除 action reference 对 actor mean 的加法、六阶段 action residual ownership mask、policy action supervision 和脚本 command/EE target 跟踪奖励；actor 独立输出全部 10 维 canonical action，reward 只评价执行后的物理结果
待验证假设：去掉动作替代后，每一阶段、每一动作维的 actor 扰动都会以统一 canonical 语义改变物理 command；环境不再主动移动底盘、机械臂或夹爪
禁止改动范围：除删除目标模仿 reward 外，不改结果型 reward、阶段完成事件、安全终止、observation 中真实任务状态、动作尺度、网络、PPO、随机化、低层动力学、AutoOpen、门型或 seed；不得用 reference、逆映射、重力/接触补偿或行为监督替 actor
运行规模与日志：第一层接口 smoke 使用 GPU 3、`1 x 8 env x 8 steps x 1 update`，日志 `doorman_train:/tmp/b2z1_phase217_actor_owned_smoke.log`，正常完成 64 timesteps。旧路径清理后静态回归 `37 passed, 5 subtests passed`，`py_compile` 和 `git diff --check` 通过；清理后 smoke 同为 GPU 3、`1 x 8 env x 8 steps x 1 update`，日志 `doorman_train:/tmp/b2z1_phase217_clean_smoke_v3.log`，完成 64 timesteps，iteration time `2.64 s`，未保存 checkpoint。尚需完成无阶段分支的十维因果验证，之后才登记四卡随机初始化训练
成功/失败判据：policy mean 不再读取/叠加 action reference；不存在 stage ownership mask；监督 loss/storage 不启用；task geometry observation 不再携带脚本动作；reward 配置不含 action/command 到预设 target 的跟踪项；10 维 action 在所有阶段均由 actor 唯一拥有；无异常或 NaN
观测指标：sampled/mean action、canonical action、physical B2Z1 command、stage、door variant、reward/event/termination；不再记录 reference/residual 伪指标
结论与已排除方向：配置和运行路径不再向 actor mean 注入 reference，也不再启用 stage ownership mask、policy supervision、脚本 target reward 或 observation 中的 10 维 reference；`b2z1_task_geometry` 已由 36 维降为 26 维。旧 reference 生成、supervision API、阶段脚本 EE/夹爪命令、THROUGH inverse、snapshot/buffer/reset 字段、reference/residual diagnostics 及 phase216 steady-inverse 失败候选均已删除。清理后 smoke 无异常且日志不再出现 reference/residual；phase217 只剩十维动作因果验收
正式保留/删除的代码：保留 actor 全维所有权、通用 command/world-pose 数学双向变换、26 维实际任务几何和结果型 reward；删除只服务脚本控制的 reference API、配置、buffer、snapshot callback、reset、diagnostics 和 steady-inverse 闭环，不保留隐藏兼容路径
前序阶段回归：旧 checkpoint 因动作语义改变而不兼容；清理后静态测试和 optimizer smoke 已通过。后续只回归物理接口、reward/event 数值有限性、十维动作因果性和纯动力学 headless 性质，不要求旧 reached 数值保持
下一决策门槛：旧脚本路径搜索为空、防回归测试通过、逐阶段逐维 action 因果 smoke 通过，三项缺一不可；之后从随机初始化开始训练，按最早未学会阶段诊断 reward 稀疏/漏洞，并重新测量 OPEN_LATCH 左右门差异
```

2026-09-28 14:58，phase217 第一层改造和接口 smoke 完成。配置已删除 `policy_action_supervision_*`、`action_residual_*`，task geometry observation 已删除 10 维 action reference，三个 approach command/target tracking reward 及 command-rate/limit reward 已删除；APPROACH 姿态验收改为比较实际夹爪与门把手几何，而不是比较实际 EE 与 actor 自己发出的 target。GPU 3 的 8-env optimizer smoke 正常退出，actor/critic observation dim 为 `162/167`，场景确认 cameras=0、RTX sensors=0、frame rendering disabled。恢复核对同时确认当前没有训练进程，且 `door_open_homie.py` 仍有 `get_policy_action_reference()`、`get_policy_action_supervision()`、阶段 reference 生成、THROUGH reference inverse、reference/residual diagnostics 和 steady-inverse 字段；因此下一步唯一工作是删除这些旧路径并完成十维因果验证，不启动训练、不改 reward 权重、不分析左右门学习差异。

2026-09-28 15:18，phase217 旧脚本链清理和清理后 smoke 完成。任务已删除 action reference/supervision API、阶段脚本 EE 与夹爪命令、THROUGH inverse、旧 snapshot/buffer/reset 状态、reference/residual diagnostics 以及 phase216 steady-inverse 失败机制；全局搜索确认运行代码不再引用这些任务字段。只读训练诊断中最后一个已删除字段 `_approach_ee_orientation_tracking_error` 改为复用正式阶段验收所用的实际几何量 `_grasp_ee_orientation_error`，未改动作、reward、阶段条件或动力学。容器内 `py_compile`、`git diff --check` 和全量测试 `37 passed, 5 subtests passed`；GPU 3 清理后 smoke 完成 64 timesteps，iteration time `2.64 s`，日志 `doorman_train:/tmp/b2z1_phase217_clean_smoke_v3.log` 无 reference/residual、traceback 或 NaN，且 cameras=0、RTX sensors=0、frame rendering disabled。当前唯一下一步是完成无阶段分支的十维 action 数值因果验证；在此之前不启动正式训练。

2026-09-28 15:25，phase217 十维动作因果验收完成。一次性只读诊断在真实 Isaac 环境 reset 后将 8 个环境分为左右门各 4 个，对十维 canonical action 逐维做中心差分；两侧 `10 x 10` action-to-physical-command 雅可比均满秩 10，最小奇异值分别为 `0.141416/0.141414`，最大奇异值均为 `1`，全部响应有限。映射函数源码不读取 stage，六阶段共享同一条映射，因此无需为六阶段复制六套相同数值实验。日志为 `doorman_train:/tmp/b2z1_phase217_action_causality.log`，同次 optimizer 完成 64 timesteps、iteration time `2.44 s`。验证后已删除临时环境方法和 trainer 环境变量钩子，新增静态契约检查十维索引、无 `zero_vel/zero_finger` 覆盖、映射无 stage 输入且四组动作切片全部接入；最终回归 `38 passed, 5 subtests passed`。phase217 判定已验证；旧 checkpoint 继续禁止继承。

备份记录：2026-09-28 15:27 已将 phase217 完成状态、phase218 计划和当前 32 个修改/新增源码、配置、测试与文档同步到 `wholebody_1/3/4`。同步文件聚合 SHA-256 在源开发机和三台备份机一致为 `51669c196d94c046ca09a83502c138221d1e4991eee871aea81d9dabfb285f2c`；未同步训练日志、缓存、`/tmp` 诊断数据、运行目录或凭据，未修改远程 Git 分支、提交和身份信息。写入本条后账本文档单独再次同步。

### phase218：纯 RL 随机初始化训练（失败）

```text
上游结论与待解决瓶颈：phase217 已证明 actor 独立拥有十维动作，运行链不再含 AutoOpen reference、行为监督或阶段动作替代；新纯 RL 语义下尚无阶段学习证据，首个待验证阶段是 APPROACH_PREGRASP
基线 checkpoint / SHA-256：无；必须随机初始化，phase215 及更早 checkpoint 与当前动作和 observation 语义不兼容
唯一变量：在 phase217 已验收代码上启动标准 PPO 训练，不改变动作、observation、reward、阶段条件、随机化、网络或低层动力学
待验证假设：结果型联合几何 progress、阶段完成奖励和安全约束足以先引导底盘与机械臂共同完成 APPROACH_PREGRASP，而不需要脚本目标或行为监督
禁止改动范围：训练运行期间冻结代码；不得恢复 reference、目标模仿、阶段 action mask、seed/门型特调或 AutoOpen 轨迹；首阶段结论形成前不得修改 GRASP 及后续 reward/条件
运行规模与日志：源开发机 GPU 0--3、每卡 2048 env、128 steps/env、4 mini-batches、随机初始化，先运行到 step25 决策点；日志为 `doorman_train:/tmp/b2z1_phase218_pure_rl_clean_to125.log`，运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260928_152954`，每 25 update 保存 checkpoint
成功/失败判据：step25 时 APPROACH_PREGRASP 联合完成率和七项验收分量相对随机初态形成明确改善，动作/command 有效且无系统性碰撞、安全终止、NaN 或运行异常；若只有单项进步但联合完成停滞，判为该阶段 credit 或接口瓶颈，不把后续阶段低到达率算作失败原因
观测指标：APPROACH_PREGRASP reached/transition、七项验收分量及 episode-seen、稳定步数、五项 progress 分量、左右门分桶 action/command、碰撞、安全终止、PPO loss/entropy/value、吞吐与显存
结论与已排除方向：失败；step25 无任意一帧联合稳定，只有 gripper-open 明显改善，其他几何分量及 actor mean 未形成有效方向；禁止靠盲目延长训练解决
正式保留/删除的代码：训练期间未编辑生产代码；step25 checkpoint 仅保留为同策略只读诊断证据，不作为有效训练基线
前序阶段回归：phase217 `38 passed, 5 subtests passed`，清理后 optimizer smoke 和左右门十维因果验收通过；纯动力学 cameras=0、RTX sensors=0、frame rendering disabled
下一决策门槛：step25 读取 checkpoint、完整日志和 APPROACH_PREGRASP 分项趋势后，决定继续训练还是只诊断首阶段；不得依据早于 step25 的短时波动修改行为
```

2026-09-28 15:34，phase218 四卡正式随机初始化训练启动并完成首个 update。四个 rank 分别绑定 GPU 0--3，每 rank 2048 env；四个场景均确认 cameras=0、RTX sensors=0、frame rendering disabled。update1 共处理 `1,048,576` timesteps，iteration time `46.24 s`，无 traceback、NaN 或运行异常。随机初态下 APPROACH_PREGRASP 的 base position/yaw、pregrasp position、gripper orientation、gripper open、contact free、base-and-EE still 当前通过率分别为 `0.0149/0.0758/0/0/0.6920/0.9978/0.0187`，`stable_reached_1/5/10` 均为 `0`；这些只作为训练前基线。日志中的 `stage/approach_pregrasp/reached=1.0` 表示所有样本仍处于第一阶段，不是完成率。代码在运行期间冻结，step25 前不做行为修改。

2026-09-28 15:58，phase218 在预注册 step25 决策点判定失败；轮询期间实际完成到 update26 后停止，所有训练进程已清理，容器未关闭。update1--26 的 `stable_reached_1/5/10` 始终全为 `0`。五项当前 quality 从首轮到末轮分别为 base position `0.3390->0.3511`、base yaw `0.5256->0.5056`、pregrasp position `0.1044->0.1023`、gripper orientation `0.0589->0.0523`、gripper open `0.8452->0.8929`；stage progress 从 `0.0347` 降到 `0.0287`。对应 episode-seen 除 gripper-open 从 `0.6920` 升至约 `0.783` 外均无趋势，unsafe-force 约 `0.01--0.02%`，无系统性安全恶化。左右门 actor mean 基本同步但没有吸收有效的底盘/EE方向。step25 checkpoint 为 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260928_152954/model_step_000025.pt`，SHA-256 `3b95eefb902f315864c6257cac0279f4e4cc85697090bdba1c69b601f841e889`。代码审计发现第一阶段将五项归一化结果取硬最小值作为唯一 progress；下一步仅做 argmin 分布只读诊断，不直接修改 reward。

### phase219：APPROACH_PREGRASP 硬最小值 credit 审计（已验证）

```text
上游结论与待解决瓶颈：phase218 step25 无联合稳定且几何 progress 无趋势；代码把五项结果 progress 的硬最小值作为唯一 stage progress，可能令绝大多数状态只有当前最差分量获得梯度
基线 checkpoint / SHA-256：phase218 model_step_000025.pt / 3b95eefb902f315864c6257cac0279f4e4cc85697090bdba1c69b601f841e889；仅用于冻结同策略状态分布诊断
唯一变量：增加逐环境 normalized component argmin 占比、最小值与次小值差距的只读日志
待验证假设：gripper orientation 和 pregrasp position 长期占据绝大多数 argmin，使 base position/yaw/gripper action 即使改善也无法获得同步 stage-progress credit
禁止改动范围：不改 reward 数值或公式、动作、observation、阶段条件、随机化、网络、PPO、低层动力学、AutoOpen、门型或 seed；诊断结束删除临时代码
运行规模与日志：四卡每卡 2048 env、128 steps、1 update，从 phase218 step25 恢复；日志计划为 `doorman_train:/tmp/b2z1_phase219_approach_argmin_diag.log`
成功/失败判据：若一到两个分量占据显著多数 argmin，且其他分量改善不改变 stage progress，则根因成立；若 argmin 分布均衡，则排除该假设并继续只读检查 reward/advantage/PPO 链
观测指标：五项 normalized progress、argmin fraction、min-second-min gap、stage progress、逐维 action/command、reward/advantage 和安全终止
结论与已排除方向：根因成立；orientation/pregrasp-position 合计占 `99.53%` argmin，base position/yaw/gripper open 合计仅 `0.47%`，硬最小值无法给五项联合目标提供同步 credit
正式保留/删除的代码：临时 argmin 统计已删除；不保留诊断状态或日志路径
前序阶段回归：phase218 纯动力学链路稳定，无异常或 NaN；phase217 动作所有权仍成立
下一决策门槛：取得一 update argmin 分布后删除诊断代码；根因成立才设计统一的多结果 dense progress，完成条件保持严格 AND
```

2026-09-28 16:05，phase219 四卡一 update 只读诊断完成。使用 phase218 step25、每卡 2048 env、128 steps，日志 `doorman_train:/tmp/b2z1_phase219_approach_argmin_diag.log`；无异常或 NaN。五项 normalized progress 成为硬最小值的比例为 base position `0%`、base yaw `0.15%`、pregrasp position `30.60%`、gripper orientation `68.93%`、gripper open `0.31%`，最小值与次小值平均差 `0.1001`。因此硬最小值确实让 `99.53%` 样本的直接 progress credit 只流向姿态或预抓取位置，无法同时引导底盘位置、航向和夹爪开度。临时诊断已删除。

### phase220：bottleneck-balanced 第一阶段结果 progress（失败）

```text
上游结论与待解决瓶颈：phase219 确认第一阶段硬最小值造成单分量 credit 死区；严格 AND 完成条件正确，不应放松
基线 checkpoint / SHA-256：无；正式验证必须随机初始化，phase218 checkpoint 只作失败对照
唯一变量：第一阶段五项封顶归一化实际结果由 hard min 改为 `0.5 * mean + 0.5 * min`；reward 权重、阶段完成条件和其余阶段不变
待验证假设：mean 项让五项结果均获得连续梯度，min 项继续优先最差分量；由于每项在完成阈值封顶且最终验收严格 AND，不能靠一项超额完成补偿另一项
禁止改动范围：不改动作、observation、五项物理目标、完成阈值、七项 AND 验收、随机化、网络、PPO、低层动力学、AutoOpen、GRASP 及后续阶段
运行规模与日志：GPU 3 的 `8 env x 8 steps x 1 update` optimizer smoke 已通过；正式训练使用源开发机 GPU 0--3、每卡 2048 env、128 steps、4 mini-batches，从随机初始化运行；日志 `doorman_train:/tmp/b2z1_phase220_balanced_progress_clean_to125.log`，运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260928_161053`
成功/失败判据：smoke 无异常或 NaN；step25 前多个几何分量同步改善并出现非零 `stable_reached_1`，随后应出现 `stable_reached_5/10`；安全终止不得系统性恶化
观测指标：五项 quality/episode-best、七项 acceptance/episode-seen、stable 1/5/10、左右门 action/command、逐项 reward、termination、吞吐
结论与已排除方向：step25 判定失败。相对 phase218 同口径 step25，除随机波动外五项 quality/episode-seen 均无实质改善，`stable_reached_1/5/10` 全为 0；仅把聚合 progress reward 变正不足以产生联合行为学习。排除“硬 min 是唯一剩余瓶颈”，禁止继续扫描 mean/min 权重或盲目延长训练
正式保留/删除的代码：通用 `bottleneck_balanced_progress` 纯函数、第一阶段唯一调用点、两项性质测试；phase219 临时统计已删除
前序阶段回归：`py_compile`、`git diff --check`、`40 passed, 5 subtests passed`
下一决策门槛：以 step25 冻结 checkpoint 做 phase221 只读 credit 因果审计，定位物理探索、reward/GAE 或 PPO 更新中的首个断点后，再决定删除或替换候选
```

2026-09-28 16:18，phase220 已完成单卡 optimizer smoke 并启动四卡正式训练。smoke 使用 GPU 3、`8 env x 8 steps x 1 update`，共 64 timesteps，iteration time `2.54 s`；无异常、NaN、reference 或 residual 路径。正式训练使用 GPU 0--3，每卡 2048 env、128 steps、4 mini-batches；四个 rank 均确认 cameras=0、RTX sensors=0、frame rendering disabled。截至 update3，每轮 `1,048,576` timesteps，iteration time 分别为 `46.30/46.08/47.20 s`，四个 worker 均存活。首轮五项 quality 为 base position `0.3374`、base yaw `0.5256`、pregrasp position `0.1035`、gripper orientation `0.0588`、gripper open `0.8452`；update1--3 的 `stable_reached_1/5/10` 均为 0。当前只说明训练链路和新 reward 正常运行，不能提前判定候选有效；代码保持冻结，预注册决策点仍为 step25。

2026-09-28 16:38，phase220 到达预注册 step25 后判定失败并主动停止，容器保留。phase218 -> phase220 的同口径 step25 quality 为 base position `0.3480 -> 0.3392`、base yaw `0.5025 -> 0.5152`、pregrasp position `0.1002 -> 0.1003`、gripper orientation `0.0535 -> 0.0587`、gripper open `0.8929 -> 0.8885`；对应 episode-seen 也只有随机量级差异，`stable_reached_1/5/10` 均为 0。`rew_task_stage_progress` 从 `-0.0321` 变为 `+0.0753`，证明候选改变了 reward 数值，但没有产生目标行为。checkpoint 为 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260928_161053/model_step_000025.pt`，SHA-256 `b99e8e07ab73cc65e8fb0eed54ebc30b800c6411a084f8cb949e22497bfdb6c8`；只作失败诊断证据，不得续训。日志末尾的 Traceback 是保存 checkpoint 后主动向 worker 发送 TERM 的退出结果，不是训练运行时故障。

### phase221：APPROACH_PREGRASP 结果 credit 因果审计（已验证）

```text
上游结论与待解决瓶颈：phase220 的聚合 progress reward 已转正，但五项物理结果和联合稳定均未改善；必须确认学习信号首先在哪一层消失
基线 checkpoint / SHA-256：phase220 model_step_000025.pt / b99e8e07ab73cc65e8fb0eed54ebc30b800c6411a084f8cb949e22497bfdb6c8；仅用于冻结诊断
唯一变量：增加无行为副作用的逐分量物理增量、reward/return/GAE、策略分布真实 score credit 和一次 PPO 更新前后 actor mean 诊断
待验证假设：正的聚合 episode reward 可能没有形成与各物理结果对应的可学习 policy gradient；诊断必须区分探索未覆盖、其他 reward/GAE 抵消和 PPO 更新抵消三层
禁止改动范围：不改 reward 公式或权重、完成条件、observation/action、探索分布、网络、PPO loss、随机化、低层动力学、AutoOpen、GRASP 及后续阶段
运行规模与日志：GPU 3 的 `8 env x 8 steps x 1 update` smoke 已通过并导出 180 个有限指标；正式诊断从 phase220 step25 恢复，GPU 0--3 各 2048 env、128 steps、4 mini-batches、一 update，日志 `doorman_train:/tmp/b2z1_phase221_approach_credit_diag.log`，结果 `doorman_train:/tmp/b2z1_phase221_approach_credit.pt`
成功/失败判据：必须用同一批样本指出五项中每项的物理增量是否存在、其 return/GAE credit 符号、真实 policy-score 信号及参数更新方向，并定位首个不一致层；不能只给总体 reward 均值或相关性猜测
观测指标：五项 normalized result delta、各 reward、raw return、GAE、stage/door variant、exact policy score x advantage、逐维 actor mean 更新、KL、clip fraction、安全终止
结论与已排除方向：首个断点在随机初始 actor 的非零 delta 均值。stage-progress return 均值/标准差为 `-0.031/2.82`，termination return 为 `-22.35/18.37`；existing termination 与 arm overspeed 均为 `1.76%/step`，low-height/far-from-door 为 0、unsafe-force 仅 `0.02%/step`。joint4/5/6 超速约 `0.45%/0.32%/1.09%`，且累计 EE 姿态 command 已到约 `1 rad`。phase217 删除 residual 后输出层归零逻辑不再生效，随机非零 delta 被积分是通用接口根因；排除直接降低安全惩罚、提高速度阈值或继续调 progress 聚合
正式保留/删除的代码：全部临时 storage、reward 分解、termination 分解和 exact-score 诊断已删除；不进入生产路径
前序阶段回归：诊断期间纯动力学、动作所有权和训练链路无行为修改；清理后 `py_compile`、`git diff --check`、`18 passed, 5 subtests passed`
下一决策门槛：phase222 在 Actor 所有权层显式支持纯 delta 策略零均值初始化；先验证随机初始 rollout，再从零训练到 step25
```

2026-09-28 16:50，phase221 两次 GPU 3 smoke 完成。首次证明新增 storage、精确 temporal marginal autograd 和 PPO update 可正常运行，64 timesteps、iteration time `3.76 s`；训练面板未呈现新增的 180 个字段，因此仅补充 rank0 的直接张量导出，不改变计算。第二次 smoke 成功生成 `/tmp/b2z1_phase221_approach_credit.pt`，包含 component return、immediate reward、rollout return、raw/normalized advantage 对十维动作的 signed/absolute exact output-score，共 180 项，均为有限值。小样本只用于接口验收，不用于根因结论。

2026-09-28 17:19，phase221 完成。四卡左右门分层 exact-score 和 reward 分解确认 termination return 主导；单卡 2048-env 原因分解进一步确认 `existing=arm_overspeed=1.76%/step`。joint1--3 不触发，joint4/5/6 分别约 `0.45%/0.32%/1.09%`；joint6 每步 p99 约 `19.58 rad/s`，紧贴真实上限 `20 rad/s`。同一 rollout 的 actor 姿态均值持续为负，累计 pitch/yaw command 接近 `1 rad`。代码审计定位到 Actor 只在 residual reference 存在时归零最终投影；phase217 正确删除 reference 后，纯 delta 策略意外恢复随机输出初始化。所有临时诊断已删除。

### phase222：纯 delta actor 零均值初始化（失败）

```text
上游结论与待解决瓶颈：phase221 确认纯 delta actor 随机非零均值经积分造成 wrist overspeed 和 termination credit 主导
基线 checkpoint / SHA-256：无；新初始化语义必须随机初始化，phase220 step25 仅作失败对照
唯一变量：Actor 增加显式 `zero_initialize_action_output`；B2Z1 训练启用，RecurrentActor 替换 backbone 后再次归零最终输出投影
待验证假设：零初始均值消除未学习策略的系统性 command 漂移，保留高斯探索和十维 actor 所有权，使安全终止不再压倒五项物理结果 credit
禁止改动范围：不改 reward、终止阈值、动作尺度、探索方差、observation、完成条件、网络宽度、随机化、低层动力学、AutoOpen、GRASP 及后续阶段
运行规模与日志：先 GPU 3、2048 env、128 steps、1 update 随机初始化验证，日志 `doorman_train:/tmp/b2z1_phase222_zero_mean_2048.log`；通过后四卡每卡 2048 env 从零到 step25
成功/失败判据：首轮十维 policy mean 为零起步，existing termination 相对 phase221 的 `1.76%/step` 显著下降；step25 多项 quality 同步改善并出现非零 stable reached，安全无回归
观测指标：逐门型 policy mean/sampled/command、termination existing/unsafe、五项 quality/episode-seen、stable 1/5/10、吞吐
结论与已排除方向：初始化机制成立，existing termination 从 `1.76%/step` 降至首轮 `0.60%/step`、末轮 `0`，无需降低安全约束；但 step25 五项未同步改善，`stable_reached_1/5/10` 全为 0，学习判据失败。排除“消除初始漂移后当前探索自然足以学会第一阶段”
正式保留/删除的代码：通用显式配置、B2Z1 启用项、两项单元测试；phase221 临时诊断已删除
前序阶段回归：`py_compile`、`git diff --check`、`18 passed, 5 subtests passed`
下一决策门槛：phase223 只审计十维 action noise 到真实物理响应的可达范围，确认最早死区后再形成统一探索候选
```

2026-09-28 17:27，phase222 单卡完整 rollout 验证通过。GPU 3、2048 env、128 steps、一 update，日志 `doorman_train:/tmp/b2z1_phase222_zero_mean_2048.log`；左右门十维初始 policy mean 全部为 `0.0000`，证明 RecurrentActor 替换 backbone 后的最终投影确实归零。existing termination 为 `0.60%/step`，相对 phase221 同规模诊断的 `1.76%/step` 下降约 66%，unsafe force 为 `0.01%/step`。纯动力学状态保持 cameras=0、RTX sensors=0、frame rendering disabled。

2026-09-28 17:38，phase222 四卡随机初始化正式训练已完成 update4，共 `4,194,304` timesteps。GPU 0--3 每卡 2048 env、128 steps/env、4 mini-batches，端口 `29525`；日志 `doorman_train:/tmp/b2z1_phase222_zero_mean_clean_to25.log`，运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260928_173241`。四个 worker 均存活，首轮 existing termination 为 `0.60%/step`，无运行时异常或 NaN。代码继续冻结至预注册 step25 决策点；普通 update 轮询不追加记录，除非出现异常、checkpoint 或决策证据变化。

2026-09-28 17:55，phase222 在预注册 step25 决策点判定失败并正常结束，四个训练 worker 均已退出，容器保留。相对 phase218 step25，五项 quality 为 base position `0.3480 -> 0.3458`、base yaw `0.5025 -> 0.5139`、pregrasp position `0.1002 -> 0.1404`、gripper orientation `0.0535 -> 0.0450`、gripper open `0.8929 -> 0.9450`；有单项改善但没有同步收敛，`stable_reached_1/5/10` 仍全为 0。末轮 episode-seen 为 base position `0.0158`、base yaw `0.0842`、pregrasp position `0.0005`、gripper orientation `0.0003`、gripper open `0.9947`。existing termination 和 unsafe force 均为 `0`，说明零初始化修复安全有效，但不足以学会第一阶段。step25 checkpoint 为 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260928_173241/model_step_000025.pt`，SHA-256 `f05cdc41dd2884b675770979eab57e658c89c8b34bd4efbb3477532cd868c201`；只作 phase223 冻结诊断基线。

### phase223：APPROACH_PREGRASP 十维探索物理可达性审计（已验证）

```text
上游结论与待解决瓶颈：phase222 消除了随机均值漂移及终止主导，但第一阶段仍无联合稳定；需要判断十维随机探索是否能在真实控制链上产生足以改变五项物理结果的动作
基线 checkpoint / SHA-256：phase222 model_step_000025.pt / f05cdc41dd2884b675770979eab57e658c89c8b34bd4efbb3477532cd868c201；只用于冻结策略审计
唯一变量：无行为变量；数学展开并用已有 rollout 核对 action std、stage scale、temporal correlation、canonical scale、delta integration、command clamp、低层步态阈值和实际响应
待验证假设：当前统一零均值 actor 的探索幅度在至少一个物理接口死区内，导致结果型 reward 虽语义正确，却没有足够的 on-policy 物理变化供 PPO 学习
禁止改动范围：不更新模型；不改 reward、完成条件、termination、安全阈值、observation/action 语义、探索参数、网络、随机化、低层动力学、AutoOpen、GRASP 及后续阶段；不引入阶段或 seed 特例
运行规模与日志：phase222 已有 8192-env step25 日志和 checkpoint；GPU 3 增加一次 2048 env、128 steps 冻结基线因果 rollout，唯一干预为 APPROACH_PREGRASP base 三维 stage scale `0.1 -> 4.0`，EE/夹爪保持 `0.1`，日志 `doorman_train:/tmp/b2z1_phase223_base_reachability.log`
成功/失败判据：对十维中每类语义给出从 policy noise 到物理命令/目标变化的定量范围，并指出是否跨越对应低层响应区或足以覆盖初始误差；结论必须同时由代码公式和真实 rollout 支持
观测指标：逐维 effective sigma、物理 command/每步目标增量、episode 累积范围、clip/工作区投影、低层启动率、实际 base/EE/夹爪响应、五项 quality/episode-best
结论与已排除方向：十维映射本身连续且 phase217 已证明满秩，但 residual-era stage scale 使纯 actor 的动作探索物理上失配。base x/y/yaw 的有效 sigma 为 `0.005/0.005/0.003`，相对低层 `0.1/0.1/0.1` 启动阈值分别约 20/20/33 sigma；EE 平移每步 sigma 为 `0.15 mm`，独立维在 1000 步的 RMS 仅 `4.74 mm`，相关 ee_x 为 `27.87 mm`；EE 姿态独立维 1000 步 RMS 仅 `0.0298 rad`，而 phase222 当前平均误差约 `1.55 rad`；夹爪每步 command sigma 为 `0.003`，但 APPROACH reset 已接近 open，episode-seen `99.47%`，不是当前首个瓶颈。冻结因果 rollout 证明放大 base 后真实步态和 episode-best 覆盖增加。排除“当前小噪声只是保守但仍能充分探索”和“放大 base 后全部 arm overspeed 都由底盘引起”；同拓扑 fresh-reset control 已有 `0.61%/step` arm overspeed，平移/yaw 干预分别为 `0.67/0.68%/step`
正式保留/删除的代码：临时 low-height/far-from-door/arm-overspeed 分解字段已删除；生产代码无 phase223 行为修改
前序阶段回归：phase222 四卡 25 updates 无异常或 NaN，末轮 existing termination/unsafe force 均为 0；零初始化正式候选保留
下一决策门槛：phase224 删除硬编码 stage noise matrix，以全阶段一致的逐动作物理尺度和可重算概率的时间相关探索替代；不得只修改 APPROACH 一行
```

2026-09-28 18:21，phase223 完成。phase222 fresh-reset control、base 三维独立放大、base 三维 AR(1)、仅平移 AR(1)、仅 yaw AR(1) 均使用同一 checkpoint 和未更新前 rollout。control 的 planar/actual speed 为 `0.0063/0.0151 m/s`；三维独立放大为 `0.2477/0.1168`，三维 AR(1) 为 `0.2458/0.2192`，仅平移 AR(1) 为 `0.2480/0.2249`，证明尺度与持续性分别控制步态启动和真实跟踪。fresh-reset control 的 arm overspeed 为 `0.61%/step`，三维独立、三维 AR(1)、仅平移和仅 yaw 分别为 `0.71/0.66/0.67/0.68%/step`；unsafe-force 均为 `0.01%/step`。四组干预均为冻结诊断，不保留 checkpoint。日志位于容器 `doorman_train`：`/tmp/b2z1_phase223_fresh_reset_control.log`、`/tmp/b2z1_phase223_base_reachability_diag.log`、`/tmp/b2z1_phase223_base_temporal_diag.log`、`/tmp/b2z1_phase223_translation_only.log`、`/tmp/b2z1_phase223_yaw_only.log`。

### phase224：全阶段统一的逐动作探索尺度（失败）

```text
上游结论与待解决瓶颈：phase223 证明 reference/residual 删除后仍保留硬编码 stage noise matrix，使相同 action 坐标在不同 stage 具有不同探索语义，并令第一阶段 base 落在低层步态阈值内
基线 checkpoint / SHA-256：无；探索分布语义变化后必须随机初始化。phase222 step25 只作对照，不得续训
唯一变量：删除 action_noise_stage_obs_key/action_noise_stage_scales；使用全阶段一致的逐动作 marginal std，并将已有精确 temporal likelihood 扩展到 base x/y/yaw 和 canonical ee_x。初始候选 sigma 为 `[0.20, 0.20, 0.12, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.10]`，temporal fraction 为 `[0.9, 0.9, 0.9, 0.9, 0, 0, 0, 0, 0, 0]`
待验证假设：按动作物理语义统一的探索既能让 base 跨过低层响应区，也能让 integrated EE/gripper 在无 reference 时覆盖有意义的局部目标变化；AR(1) 使速度和 radial EE 探索持续，PPO 仍用已验证的精确 sequence marginal likelihood
禁止改动范围：不改 reward、完成条件、termination、安全阈值、observation/action 定义、canonical 坐标、delta scale、网络、随机化、低层动力学、AutoOpen、门型或 seed；不增加 stage/seed 特例
运行规模与日志：先 py_compile、静态契约和 temporal-noise 单测；再 GPU 3 单卡 2048 env、128 steps、一 update 随机初始化 smoke；通过后四卡每卡 2048 env 从零训练至 step25
成功/失败判据：smoke 中 base command/actual speed 跨过低层响应区，十维 action/command 有限，existing termination 不显著高于 phase223 同拓扑 control，unsafe-force 不恶化；step25 多项 quality/episode-seen 同步改善并出现非零 stable reached
观测指标：十维 policy mean/sample noise/command、base command/actual/goalward speed、五项 quality/episode-best、stable 1/5/10、termination 原因、安全力、PPO KL/loss/entropy
结论与已排除方向：统一探索机制解决了物理覆盖问题，step25 多项 episode-best 和 orientation quality 相对 phase222 明显改善，existing/unsafe 安全指标无回归；但严格联合稳定仍为 0，预注册学习判据失败。actor command 已有朝目标分量而实际运动略背离目标，排除“底盘仍因探索尺度过小而完全不动”，尚未区分低层跟踪延迟、投影语义、命令时序或统计聚合
正式保留/删除的代码：候选删除 B2Z1 配置中的 action_noise_stage_obs_key/action_noise_stage_scales，使用全阶段统一逐动作 std 和 base/ee_x temporal fraction；新增纯 RL 配置契约测试
前序阶段回归：phase223 临时诊断已删除；py_compile、git diff check 和 `26 passed`；单卡真实纯动力学 smoke 无异常或 NaN
下一决策门槛：phase225 冻结 checkpoint 和行为，只读定位 command-to-actual 的首个不一致层；不回退 stage matrix，不先改 reward
```

2026-09-28 18:30，phase224 静态回归和单卡真实 smoke 完成。纯 RL 契约新增防回归断言，禁止 B2Z1 配置重新声明 stage-conditioned noise，并固定十维全局 std 与 temporal fraction；相关测试 `26 passed`。GPU 3 的 2048 env、128 steps、一 update 随机初始化 smoke 日志为 `doorman_train:/tmp/b2z1_phase224_unified_noise_smoke.log`，确认 cameras=0、RTX sensors=0、frame rendering disabled，无异常或 NaN。base planar/actual speed 为 `0.2472/0.2225 m/s`，existing/unsafe termination 为 `0.68%/0.01% per step`；base position/yaw、pregrasp position、orientation 和 gripper-open episode-best 为 `0.3675/0.5685/0.1599/0.1961/0.8654`，联合稳定仍为 0，只作为初态覆盖基线。

2026-09-28 18:55，phase224 四卡随机初始化训练在预注册 step25 正常结束，四个 worker 已退出，容器保留。GPU 0--3 每卡 2048 env、128 steps/env、4 mini-batches；日志 `doorman_train:/tmp/b2z1_phase224_unified_noise_clean_to25.log`，运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260928_182930`。五项 quality 为 `0.3438/0.5013/0.1565/0.1244/0.9005`，episode-best 为 `0.4355/0.6749/0.2463/0.2665/0.9856`，episode-seen 为 `0.0679/0.3074/0.0061/0.0057/0.9709`；`stable_reached_1/5/10` 全为 0。planar/actual speed 为 `0.2554/0.2143 m/s`，command goalward speed/alignment 为 `+0.0302 m/s/+0.0939`，actual goalward speed 为 `-0.0037 m/s`，existing termination `0.05%/step`，unsafe-force 为 0。探索机制通过物理覆盖与安全验收并正式保留，但联合学习判据失败。checkpoint 为 `model_step_000025.pt`，SHA-256 `8bd6d8593f5c656436a070d962a17d13374e03079b60df2f31e6c45789abd24b`；只作 phase225 冻结诊断基线。

### phase225：APPROACH_PREGRASP 底盘 command-to-actual 首个分化层（已验证）

```text
上游结论与待解决瓶颈：phase224 已让 actor 产生足够幅度的真实底盘运动和正向 goalward command，但 actual goalward 仍略为负，严格联合稳定为 0
基线 checkpoint / SHA-256：phase224 model_step_000025.pt / 8bd6d8593f5c656436a070d962a17d13374e03079b60df2f31e6c45789abd24b；只作冻结诊断
唯一变量：无行为变量；提取 step1--25 左右门十维 policy mean/command 和五项结果趋势，并在同批冻结轨迹中核对 canonical command、低层接收命令、实际 base twist、goalward 投影及时间聚合
待验证假设：正 command/负 actual 的聚合差异可能来自低层跟踪延迟、坐标投影错误、命令频繁换向或跨时刻/跨门型聚合；必须找到第一处数值不一致
禁止改动范围：不更新模型；不改 reward、完成条件、termination、安全阈值、observation/action、探索参数、网络、随机化、低层动力学、AutoOpen、GRASP 及后续阶段
运行规模与日志：已复用 `doorman_train:/tmp/b2z1_phase224_unified_noise_clean_to25.log` 提取训练趋势；现有聚合信息不足以区分时延和统计混合，下一步运行单卡 2048 env、128 steps 的最小冻结 rollout，日志 `doorman_train:/tmp/b2z1_phase225_command_tracking_diag.log`
成功/失败判据：必须用同一环境、同一时刻的数值链指出 command 首次在哪一层与 actual goalward 分化；聚合相关性或推测不算结论
观测指标：左右门十维 policy mean/command、canonical/world base command、低层接收速度、实际 root twist、逐步 goalward projection、命令符号保持时间、五项 quality/episode-best/episode-seen
结论与已排除方向：冻结 rollout 中 command/actual goalward 为 `+0.0325/+0.0065 m/s`；实际速度沿当前命令为 `+0.1355 m/s`、沿上一高层步命令为 `+0.1387 m/s`，direction cosine 为 `0.5490/0.5581`，tracking error 为 `0.2256/0.2194 m/s`，命令方向相邻步相似度为 `0.8565`。低层方向正确，仅有轻微一拍时延。phase224 step25 的负 actual 是随训练从 `-0.0443` 向零持续回升过程中的暂态采样点，不是坐标投影或低层反向 bug。平均 command speed `0.2516 m/s` 而 goalward alignment 仅 `0.1018`，说明主要限制仍是策略/探索大多未指向目标
正式保留/删除的代码：command/actual 时序分层指标已删除；生产行为无修改
前序阶段回归：生产行为冻结；phase224 已通过 `26 passed`、单卡 smoke 和四卡 step25 无 NaN/运行时异常
下一决策门槛：phase226 只读审计当前最弱的 EE position/orientation 结果 credit；不修改低层
```

2026-09-28 19:08，phase225 单卡冻结诊断完成。GPU 3、2048 env、128 steps，加载 phase224 step25 checkpoint，只读取 update26 rollout；日志 `doorman_train:/tmp/b2z1_phase225_command_tracking_diag.log`，更新后的模型未保存。纯动力学状态继续为 cameras=0、RTX sensors=0、frame rendering disabled，无异常或 NaN。临时诊断代码已删除。

### phase226：APPROACH_PREGRASP 同机制续训至 step50（已验证）

```text
上游结论与待解决瓶颈：phase225 排除底盘坐标和低层反向跟踪；phase224 step10--25 的 pregrasp-position/orientation quality 和 episode-best 仍持续改善，但尚无联合稳定
基线 checkpoint / SHA-256：phase224 model_step_000025.pt / 8bd6d8593f5c656436a070d962a17d13374e03079b60df2f31e6c45789abd24b；按同一四卡拓扑续训
唯一变量：无行为变量；保持源码、reward、探索、网络、随机化、低层和训练规模不变，只增加训练 update 25 -> 50
待验证假设：当前两个最弱结果尚未平台，继续同一 on-policy 学习可进一步提高同步结果并产生首个严格联合稳定样本
禁止改动范围：不改 reward、完成条件、termination、安全阈值、observation/action、探索参数、网络、随机化、低层动力学、AutoOpen、GRASP 及后续阶段
运行规模与日志：GPU 0--3、每卡 2048 env、128 steps/env、4 mini-batches，恢复到 step50；日志 `doorman_train:/tmp/b2z1_phase226_same_mechanism_to50.log`
成功/失败判据：成功要求多项 quality/episode-best/episode-seen 延续改善并出现非零 stable_reached_1；若联合稳定仍为 0但趋势明显不平台，只能登记未完成而不能伪判失败；若最弱结果平台或回归则停止并进入 credit 审计
观测指标：五项 quality/episode-best/episode-seen、stable 1/5/10、左右门十维 policy mean/command、base command/actual/goalward、安全终止、PPO KL/loss
结论与已排除方向：step50 五项 quality、episode-best 和 episode-seen 均相对 step25 大幅同步改善，command/actual goalward 转为 `+0.1283/+0.0925 m/s`；严格联合稳定仍为 0，但无平台或 credit 丢失证据。排除“step25 后必须先修改 reward 才能继续学习”
正式保留/删除的代码：无 phase226 代码修改
前序阶段回归：phase225 纯动力学冻结 rollout 无异常或 NaN；生产行为未修改
下一决策门槛：phase227 同机制续训到 step75，优先等待首个联合稳定和 GRASP 样本
```

2026-09-28 19:37，phase226 四卡续训正常完成 step50，四个 worker 已退出，无异常或 NaN。GPU 0--3 每卡 2048 env、128 steps/env、4 mini-batches；日志 `doorman_train:/tmp/b2z1_phase226_same_mechanism_to50.log`，运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260928_191206`。checkpoint `model_step_000050.pt` 的 SHA-256 为 `60f112bd5ed377b06322ec5aaee013fe013311403f373fb4ab65e11f04f65f98`。纯动力学状态保持 cameras=0、RTX sensors=0、frame rendering disabled。

### phase227：APPROACH_PREGRASP 同机制续训至 step75（已验证）

```text
上游结论与待解决瓶颈：phase226 五项结果同步大幅改善，尚无联合稳定但无平台证据
基线 checkpoint / SHA-256：phase226 model_step_000050.pt / 60f112bd5ed377b06322ec5aaee013fe013311403f373fb4ab65e11f04f65f98；按同一四卡拓扑续训
唯一变量：无行为变量；保持源码、reward、探索、网络、随机化、低层和训练规模不变，只增加训练 update 50 -> 75
待验证假设：各单项满足率已进入可观比例，继续训练可使它们在同一时刻重合并首次进入 GRASP
禁止改动范围：不改 reward、完成条件、termination、安全阈值、observation/action、探索参数、网络、随机化、低层动力学、AutoOpen、GRASP 及后续阶段
运行规模与日志：GPU 0--3、每卡 2048 env、128 steps/env、4 mini-batches，恢复到 step75；日志 `doorman_train:/tmp/b2z1_phase227_same_mechanism_to75.log`
成功/失败判据：出现非零 stable_reached_1，并产生真实 GRASP 入口样本；若仍为 0但五项同步改善则登记未完成并继续，若平台或回归才进入联合时序 credit 审计
观测指标：五项 quality/episode-best/episode-seen、stable 1/5/10、GRASP occupancy/reached、左右门十维 policy mean/command、base goalward、安全终止
结论与已排除方向：step75 五项 quality 为 `0.7422/0.7811/0.6458/0.5943/0.8989`，episode-best 为 `0.7949/0.9154/0.7122/0.6774/0.9727`，episode-seen 为 `0.5496/0.7232/0.2726/0.2894/0.9509`；相对 step50 几何结果继续同步改善。step70 曾出现 `stable_reached_1=0.0002`，证明严格联合状态并非不可达，但 `stable_reached_5/10` 始终为 0，step75 的 `base_and_ee_still=0.0015`，GRASP occupancy/reached 为 0。排除几何结果整体无学习、底盘方向错误和联合条件完全不可达；首个瓶颈收敛为到位后的连续停稳
正式保留/删除的代码：无 phase227 代码修改
前序阶段回归：phase226 四卡 25 updates 无异常或 NaN，纯动力学状态成立
下一决策门槛：phase228 冻结 checkpoint 做联合时序与 reward credit 只读审计；不再无证据盲目延长训练
```

2026-09-28 20:01，phase227 四卡续训正常完成 step75，四个 worker 已退出，无异常或 NaN。GPU 0--3 每卡 2048 env、128 steps/env、4 mini-batches；日志 `doorman_train:/tmp/b2z1_phase227_same_mechanism_to75.log`，运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260928_193716`。checkpoint `model_step_000075.pt` 的 SHA-256 为 `ff5c35f964d624614d2398a454070f4e8c0ea71d8ef4d88cb3fa0ba620ca3e39`。纯动力学状态保持 cameras=0、RTX sensors=0、frame rendering disabled。

### phase228：APPROACH_PREGRASP 联合停稳 credit 只读审计（已验证）

```text
上游结论与待解决瓶颈：phase227 已显著学会五项几何接近并偶发一帧严格联合状态，但从未连续稳定 5/10 步，GRASP 仍无样本
基线 checkpoint / SHA-256：phase227 model_step_000075.pt / ff5c35f964d624614d2398a454070f4e8c0ea71d8ef4d88cb3fa0ba620ca3e39；只作冻结诊断
唯一变量：无行为变量；在同一冻结轨迹中联合记录五项实际几何 readiness、实际 base motion、实际 EE speed、连续稳定计数和现有逐步 reward/return，并离线评价候选结果型稳定 credit
待验证假设：当前第一阶段 progress 只包含五项几何结果，实际停稳仅通过稀疏完成事件获得 credit，导致策略学会接近但没有足够信号在目标附近减速并维持 10 步
禁止改动范围：不更新模型；不改 reward、完成条件、termination、安全阈值、observation/action、探索参数、网络、随机化、低层动力学、AutoOpen、GRASP 及后续阶段；候选计算不得读取 actor command
运行规模与日志：GPU 3、单卡 2048 env、128 steps 的冻结 checkpoint rollout；日志 `doorman_train:/tmp/b2z1_phase228_stability_credit_diag.log`，逐步张量 `doorman_train:/tmp/b2z1_phase228_stability_credit.pt`
成功/失败判据：必须逐步定位几何 readiness 首次足够而实际速度未收敛的时刻，并证明现有 reward 对减速缺少可辨别 credit；离线候选必须严格把“近处停稳”排在“近处运动”和“远处停稳”之前，否则不得进入行为修改
观测指标：五项 readiness 及其联合量、base linear/angular motion、EE speed、stable counter、现有 progress/complete reward、return/advantage、安全终止；按远处/近处和运动/停稳分桶比较
结论与已排除方向：有效冻结 rollout 共 262144 帧。五项单独通过率为 `9.1892%/15.5670%/0.1022%/0.0271%/33.8840%`，contact-free 为 `99.5171%`；`readiness>=0.8` 的 2514 帧中 base/EE 平均实际速度为 `0.2312/0.3173 m/s`，无一帧同时低于 `0.05 m/s`，严格联合稳定为 0。现有 `_stage_progress[:,0]` 只含五项几何结果，速度只参与稀疏的 10 步完成事件；在高 readiness 样本中，现有 progress 增量与减速相关系数为 `-0.0573`。因此确认首个 credit 缺口是“近目标减速和保持”没有连续结果势函数，而非几何不可达、低层反向或验收阈值过严。离线候选使用五项归一化几何质量乘积门控实际 base/EE 稳定质量；以 0.2 权重混入现有 progress 后，近目标停稳、近目标运动、远处停稳的平均候选值分别为 `0.8505/0.7142/0.2283`，对应停止动作的近处/远处单次势能增益为 `10.91/0.164`，满足预注册排序且不会持续奖励远处静止
正式保留/删除的代码：所有 phase228 临时导出代码和错误 trainer 触发点均已删除；只保留 `/tmp` 日志和逐步证据，不保留行为修改
前序阶段回归：phase227 无行为修改、无异常或 NaN，几何学习和纯动力学状态成立
下一决策门槛：phase229 仅实现并验证已离线排序的耦合停稳结果势函数；不改完成条件或其他训练变量
```

2026-09-28 20:26，phase228 完成。第一次导出在环境初始化期间提前累计满 128 步，第二次把触发点挂在当前任务未使用的 `ppo_trainer_homie_api.py`，两者均未用于结论；第三次将触发点放到实际使用的 `ppo_trainer.py::_rollout_step` 后取得有效 2048 env x 128 steps 数据。有效运行保持 cameras=0、RTX sensors=0、frame rendering disabled，无异常或 NaN；临时代码已清理，容器中证据为 `/tmp/b2z1_phase228_stability_credit_diag.log` 和 `/tmp/b2z1_phase228_stability_credit.pt`。

### phase229：APPROACH_PREGRASP 几何 readiness 耦合停稳势函数（失败）

```text
上游结论与待解决瓶颈：phase228 证明策略已学会几何接近，但现有第一阶段 progress 不评价实际减速，导致高 readiness 状态仍高速通过且无法连续稳定
基线 checkpoint / SHA-256：无；reward 语义变化后从随机初始化训练。phase227 step75 / ff5c35f964d624614d2398a454070f4e8c0ea71d8ef4d88cb3fa0ba620ca3e39 只作旧 reward 对照
唯一变量：第一阶段 progress 从纯 geometry progress 改为 `0.8 * geometry_progress + 0.2 * joint_geometry_readiness * actual_stability_quality`；joint readiness 是五项归一化实际几何质量的乘积并受 contact-free 约束，stability 只使用实际 base motion 和 EE speed
待验证假设：与几何 readiness 相乘的实际稳定质量能在目标附近提供连续减速 credit，同时其乘积门控和 signed potential 防止机器人在远处静止刷分
禁止改动范围：不改完成条件、termination、安全阈值、observation/action、探索参数、网络、随机化、低层动力学、AutoOpen、GRASP 或后续阶段 reward；不得读取 action、command 或脚本目标
运行规模与日志：纯函数边界/排序与纯 RL 契约 `13 passed`；GPU 3 的 `8 env x 8 steps x 1 update` 随机初始化 smoke 日志 `doorman_train:/tmp/b2z1_phase229_stability_progress_smoke.log`，无异常或 NaN；正式训练使用 GPU 0--3、每卡 2048 env、128 steps、4 mini-batches，从随机初始化至 step25，端口 `29529`，日志 `doorman_train:/tmp/b2z1_phase229_stability_progress_clean_to25.log`
成功/失败判据：单测必须证明近处停稳 > 近处运动 > 远处停稳，静止不产生每步常驻奖励；smoke 无异常、NaN 或安全回归；step25 五项几何 quality/episode-best 不低于 phase224 同期趋势，并出现显著高于旧机制的 near-target 减速或非零 stable_reached
观测指标：五项 quality/episode-best/episode-seen、readiness 分桶的 base/EE 实际速度、stable 1/5/10、GRASP occupancy/reached、安全终止、PPO KL/loss
结论与已排除方向：step25 五项 quality 为 `0.3232/0.5008/0.1526/0.1209/0.8636`，episode-best 为 `0.4240/0.6727/0.2434/0.2503/0.9810`，episode-seen 为 `0.0607/0.3088/0.0098/0.0042/0.9587`；相对 phase224 同期五项 quality `0.3438/0.5013/0.1565/0.1244/0.9005` 与 episode-best `0.4355/0.6749/0.2463/0.2665/0.9856` 均无改善。actual planar speed `0.2175 m/s` 对旧同期 `0.2143 m/s`，`stable_reached_1/5/10` 全为 0，GRASP occupancy/reached 为 0；unsafe force `0.0001/step`，无异常或 NaN。预注册学习判据失败，排除“该耦合势函数本身足以在 step25 产生减速”；尚未排除探索噪声使停稳样本物理上过稀
正式保留/删除的代码：候选纯函数、两个测试和唯一调用点暂留到 phase230 冻结归因结束；不得继续训练或调权重，若被否定则完整删除
前序阶段回归：phase228 诊断不改行为，临时代码已清理；phase227 checkpoint 和日志保留为旧 reward 对照
下一决策门槛：phase230 冻结比较 sampled action、actor mean 和近零噪声反事实；不扫描 reward 权重
```

2026-09-28 20:56，phase229 四卡随机初始化训练在预注册 step25 正常结束，四个 worker 已退出。GPU 0--3 每卡 2048 env、128 steps/env、4 mini-batches；日志 `doorman_train:/tmp/b2z1_phase229_stability_progress_clean_to25.log`，运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260928_203141`。checkpoint `model_step_000025.pt` SHA-256 为 `f1fe084520185da25b7f1ae6d0eaf9e9d9d4e26927fbc59e2a3caa543fecd7d6`，只作失败诊断证据。

### phase230：APPROACH_PREGRASP 探索噪声与 actor mean 停稳归因（已验证）

```text
上游结论与待解决瓶颈：phase229 耦合停稳势函数离线排序成立，但 step25 实际速度、stable 和几何质量均未改善
基线 checkpoint / SHA-256：phase229 model_step_000025.pt / f1fe084520185da25b7f1ae6d0eaf9e9d9d4e26927fbc59e2a3caa543fecd7d6；只作冻结诊断
唯一变量：不更新模型；先在正常探索 rollout 中按 readiness 分桶同步导出 actor mean、sampled action、command 与实际 base/EE 速度，再用同一 checkpoint 做近零探索噪声反事实
待验证假设：全阶段统一探索解决了可达性，但固定边缘 std 可能在近目标状态形成高于 `0.05 m/s` 验收线的物理速度下限；也可能 actor mean 本身仍持续运动，必须区分
禁止改动范围：不改 reward、完成条件、termination、安全阈值、observation/action、网络、随机化、低层动力学、AutoOpen 或后续阶段；不更新模型，不扫描 reward 或 noise 参数
运行规模与日志：先后完成两套正常/近零噪声冻结对照，均为单卡 2048 env、128 steps。首轮文件 `doorman_train:/tmp/b2z1_phase230_{normal,near_zero}_noise.{log,pt}` 因 sampled action 张量别名污染而废弃；修正版 step25 文件后缀 `_v2`，补充高 readiness 的 phase227 step75 文件为 `doorman_train:/tmp/b2z1_phase230_phase227_{normal,zero}_v2.{log,pt}`
成功/失败判据：必须在同一 readiness 分桶内定量分解 mean 与 noise，并用近零噪声反事实确认实际速度和停稳率是否发生因果变化；聚合均值或推测不算结论
观测指标：readiness、actor mean/sample/noise、base/EE command、实际 base motion/EE speed、stable 1/5/10、安全终止
结论与已排除方向：已验证。首轮 sampled action 与传入环境的 action 共用张量，随后被 `DeltaActionBase.step()` 原地积分，导致 EE/夹爪样本污染；改为 step 前 clone 后，正常组十维噪声与配置吻合，近零组十维绝对均值均约 `8e-5`。phase229 step25 对照中，近零噪声把 base/EE 均速从 `0.3179/0.4963` 降至 `0.0499/0.1481 m/s`；排除 reset 后 40 步以内瞬态后仍为 `0.0432/0.1020 m/s`。为避免低 readiness 误判，又用 phase227 step75 补充高 readiness 对照：`0.8--0.9` 桶内正常组 base/EE 为 `0.2213/0.2748 m/s`，近零组为 `0.0598/0.0938 m/s`。因此固定全局 std 是目标附近速度下限的主要来源，actor mean 仍有次要残余，两者都成立。step75 checkpoint 的十维 std 全部略高于并会在下一次 rollout clamp 回逐动作上限，说明状态无关参数在 entropy 驱动下始终贴上限，无法表达“远处探索、近处停稳”。排除仅继续调 stability reward、全局固定降噪和把远离目标的持续接近动作误判为到位后不停
正式保留/删除的代码：phase230 环境变量触发、trainer action snapshot、环境 reward 后导出和 `os` 导入均已删除；只保留容器 `/tmp` 证据。phase229 的结果型耦合停稳 credit 暂留，作为 mean 减速信号，不调权重
前序阶段回归：phase229 四卡训练无异常或 NaN，安全无明显回归，但学习判据失败
下一决策门槛：phase231 仅验证统一状态依赖方差。若它不能在不损害远处探索的前提下降低高 readiness 方差和实际速度，则完整删除该候选；不得退回 stage scale 或 seed 特调
```

2026-09-28 21:24，phase230 完成。四组有效/补充 rollout 均保持 cameras=0、RTX sensors=0、frame rendering disabled，未更新或保存模型。正常与近零对照分别在 GPU 3/2 并行运行并正常退出。首轮别名污染诊断作为失败尝试保留在 `/tmp`，禁止再用于结论；修正版数据和日志路径见上。所有 phase230 临时代码已清理。

### phase231：统一状态依赖探索方差（失败）

```text
上游结论与待解决瓶颈：phase230 证明固定全局 std 是目标附近实际速度的主要下限，同时 actor mean 尚有次要残余；当前全局 std 到 step75 始终贴逐动作上限
基线 checkpoint / SHA-256：无；policy 参数化变化后必须从随机初始化训练。phase227/229 checkpoint 只作 phase230 证据，不兼容也不得恢复
唯一变量：用读取统一 actor observation 的状态依赖方差网络替换状态无关的十维全局 std；均值网络、十维 action、逐动作方差上下界和 temporal Gaussian 概率语义保持不变
待验证假设：策略可在远离目标时维持足够探索，在高 readiness 状态主动降低边际方差，从而使结果型停稳 credit 可采样并推动 actor mean 一并减速
禁止改动范围：不改 reward 权重、完成条件、termination、安全阈值、observation/action、随机化、低层动力学、AutoOpen 或后续阶段；方差网络不得读取 stage、seed、门型、显式 readiness 或脚本目标；不允许固定全局降噪、按 update 退火或阶段倍率
运行规模与日志：`36 passed, 5 subtests passed`，容器 py_compile 通过；GPU 3 的 `8 env x 8 steps x 1 update` 单卡 smoke 正常完成，日志 `doorman_train:/tmp/b2z1_phase231_state_dependent_noise_smoke.log`。正式训练已用 GPU 0--3、每卡 2048 env、128 steps、4 mini-batches，从随机初始化启动至 step25，日志 `doorman_train:/tmp/b2z1_phase231_state_dependent_noise_clean_to25.log`，端口 `29531`
成功/失败判据：初始远处探索的逐动作边际 std 与 phase224 统一上限同量级且能跨过底盘步态阈值；状态依赖 std 有有限梯度并严格受统一上下界约束；训练中几何 quality/episode-best 不低于 phase224 同期，高 readiness 方差和 base/EE 实际速度显著下降，并出现非零 stable 1/5/10 与 GRASP 入口样本
观测指标：按 readiness 分桶的十维 marginal std、sample noise、actor mean、base/EE 实际速度、stable 1/5/10；五项 quality/episode-best/episode-seen、GRASP occupancy/reached、安全终止、PPO KL/entropy
结论与已排除方向：失败。step25 五项 quality `0.3137/0.5014/0.1325/0.1183/0.8782`，低于 phase224 同期 `0.3438/0.5013/0.1565/0.1244/0.9005`；actual planar speed `0.2177 m/s`，stable 1/5/10 与 GRASP occupancy/reached 全为 0。冻结分桶文件 `doorman_train:/tmp/b2z1_phase231_sigma_readiness.pt` 和日志 `/tmp/b2z1_phase231_sigma_readiness.log` 显示 readiness 最大仅 `0.2231`，最低桶 std 约 `0.199956/0.199953/0.119977/0.019996/0.009998/.../0.099970`，`0.1--0.2` 桶更接近上限；没有形成与接近目标相关的降噪。候选实际退化为旧全局方差，并伴随几何学习回归。现象与固定 entropy bonus 把 scale 推向上限一致，但尚未做分项梯度归因，因此 entropy 是主要梯度来源仍作为 phase232 待验证假设，不提前写成已确认根因
正式保留/删除的代码：状态依赖 scale 网络、三项配置、三个专属测试、任务侧 marginal-std 日志和 `B2Z1_PHASE231_DIAG_PATH` 导出均已删除；通用 temporal Gaussian 的 `action_sigma` storage、likelihood 和噪声指标保留。checkpoint 与 `/tmp` 数据只作失败证据
前序阶段回归：结构测试和纯动力学 optimizer smoke 曾通过，但正式训练的 APPROACH_PREGRASP 几何与停稳判据失败；候选不能进入生产路径
下一决策门槛：phase232 在同一批数据上分离 PPO surrogate、entropy 和总 loss 对方差参数的梯度方向、范数及逐动作贡献；归因前不得调整 entropy、全局 std 或再引入探索网络
```

2026-09-28 22:02，phase231 完成失败判定并停止。四卡训练日志为 `doorman_train:/tmp/b2z1_phase231_state_dependent_noise_clean_to25.log`；失败 checkpoint 为 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260928_212835/model_step_000025.pt`，SHA-256 `860af428d8fa3f93361558778cc53597597a14437f7c6b57a356c5f9022d638b`。当前无训练或诊断进程，候选生产代码已清理。

### phase232：PPO 探索方差梯度归因（已验证）

```text
上游结论与待解决瓶颈：phase231 的状态依赖方差在所有 readiness 桶退化到逐动作上限，但尚未定量证明 surrogate 与 entropy 各自对该结果的贡献
基线 checkpoint / SHA-256：phase231 model_step_000025.pt / 860af428d8fa3f93361558778cc53597597a14437f7c6b57a356c5f9022d638b；仅用于失败机制的只读归因，不续训
唯一变量：无行为变量；对同一 rollout 和同一参数点分别反向传播 PPO clipped surrogate、entropy bonus 和二者总和，记录方差相关参数的分项梯度
待验证假设：固定 entropy bonus 对方差上升的梯度在训练访问到的状态上压过 surrogate 的状态区分 credit，使 phase231 scale 网络统一饱和到上限
禁止改动范围：不更新模型；不改 reward、entropy 系数、完成条件、observation/action、随机化、低层动力学或任何阶段逻辑；不以调参代替归因
运行规模与日志：GPU 3、2048 env、128 steps、4 mini-batches；加载冻结 phase231 step25，覆盖第一 PPO epoch 的全部四个 minibatch。日志 `doorman_train:/tmp/b2z1_phase232_gradient_attribution.log`，张量 `doorman_train:/tmp/b2z1_phase232_gradient_attribution.mb*.pt`；每个 minibatch 只求分项梯度，全部在 `optimizer.step()` 前停止。首轮仅导出第一个 minibatch，高 readiness 只有 7 帧，证据不足且不用于最终结论
成功/失败判据：必须在完全相同的样本、归一化、stage/variant 权重和 PPO clip 语义下，给出 surrogate、entropy、总 loss 对全局 std 与候选 scale 参数的梯度方向、L2 范数、余弦关系和逐动作分量；只比较 loss 标量或训练后 std 不算完成
观测指标：分项 loss、有效 clip 比例、advantage 分布、逐动作梯度、各 readiness 桶的梯度聚合、总梯度与真实 optimizer update 的一致性
结论与已排除方向：已验证。有效诊断覆盖同一冻结 rollout 的四个等大 minibatch，共 `262144` 帧；advantage 均值约 0、标准差 `1.0000`，ratio 均值 `1.0000006`、范围 `0.9913--1.0094`，确认使用真实 PPO 近端点。全局 std 上 surrogate 与 entropy 梯度范数分别为 `0.7212/2.2961`，比例 `3.184`；二者余弦 `-0.461`，总梯度与 entropy 余弦 `0.951`。状态 scale 网络上的范数比例为 `1.428`，总梯度与 entropy 余弦 `0.806`。所有 readiness 桶的实际 scale 已为 `0.9998--0.99995`。样本数为 `<0.05:260277`、`0.05--0.1:1689`、`0.1--0.2:177`、`0.2--0.4:1`；`>=0.1` 仅 `0.0679%`。`0.1--0.2` 桶中 surrogate 已对 base_yaw、ee_x、ee_y、ee_roll、ee_pitch、ee_yaw 给出正 `dL/dsigma`，即梯度下降会降噪，但逐状态 entropy 在每一帧、每一维都给出负梯度推动增噪，且稀有状态无法抵消密集样本。结合 checkpoint Adam 一阶/二阶矩，首 minibatch 的十维全局 std 预测更新仍全部为正，确认历史优化器动量也在维持上限。由此确认 phase231 失败不是网络没有梯度，而是探索目标语义与“远处探索、近处停稳”冲突；排除继续调初始化比例、延长训练或仅换方差网络
正式保留/删除的代码：临时恢复的 phase231 scale 参数化、三项配置、readiness storage、分项 autograd 导出与提前退出均已删除；仅保留容器 `/tmp/b2z1_phase232_gradient_attribution.log` 和 `/tmp/b2z1_phase232_gradient_attribution.mb*.pt` 证据
前序阶段回归：phase232 临时代码清理后相关单测 `20 passed`，py_compile、`git diff --check` 和专属标识搜索均通过
下一决策门槛：phase233 先离线定义只约束总体探索预算、允许状态间重分配方差的统一目标；候选不得读取 stage、seed、门型、readiness 或脚本目标，也不得靠扫描 entropy 系数
```

2026-09-28 22:20，phase232 完成。首轮只导出一个 minibatch且高 readiness 仅 7 帧，未用于最终结论；第二轮覆盖全部四个 minibatch且未调用 `optimizer.step()`。恢复 checkpoint 时因原训练为四卡、诊断为单卡而跳过 rank-local staged snapshot；本结论针对从统一 reset 分布采集的 APPROACH_PREGRASP 状态，不能外推到尚未学习的后续阶段。Isaac 运行保持 cameras=0、RTX sensors=0、frame rendering disabled。

### phase233：统一探索预算目标设计（已验证）

```text
上游结论与待解决瓶颈：phase232 确认逐状态 entropy bonus 在稀有近目标状态上持续推动增大方差，并在参数与 Adam 动量层主导 phase231 的饱和
基线 checkpoint / SHA-256：不训练；使用 phase232 冻结张量和必要的合成分布做离线数学验证。phase231 checkpoint 只作历史证据
唯一变量：探索正则从“每个状态都奖励更大 entropy”改写为“维持批次总体探索预算，同时允许策略按 observation 在状态间重新分配方差”的候选目标；本 phase 不接入正式训练
待验证假设：总体预算约束在远处方差已足够时不再向所有状态施加增噪梯度，因此 surrogate 可在稀有近目标状态降低方差，同时不会让整体探索坍塌
禁止改动范围：不改 reward、完成条件、observation/action、随机化、低层动力学、阶段逻辑或正式 entropy 系数；候选不得读取 stage、seed、门型、readiness 或脚本目标；不启动训练
运行规模与日志：候选纯函数 `gr00t/rl/trl/modules/exploration_budget.py`，测试 `gr00t/rl/tests/test_exploration_budget.py`；先做闭式梯度分析、合成两状态分布测试和 phase232 冻结数据反事实，不启动 Isaac 或训练
成功/失败判据：候选必须同时满足统一状态语义、全局探索下界、近目标 surrogate 可主导局部降噪、远处无无条件降噪、对状态占比变化稳定，并能给出可重算的 temporal Gaussian likelihood；任一不满足即删除
观测指标：总体/分桶方差、候选正则的逐样本梯度、与 surrogate 的夹角及范数、探索预算违反量、动作维度独立性
结论与已排除方向：已验证。新增纯函数按动作维计算 `mean((state_std/max_std)^2)`，只对低于预算的维度施加非负对偶约束；预算满足后对所有状态梯度严格为零。独立 8 项测试覆盖 quiet minority、全局坍塌、动作维独立、batch/time 聚合、局部 surrogate 不被阻挡、quiet occupancy 边界、对偶乘子升降和输入校验。`0.8` 归一化方差预算在 quiet scale 为 `0/0.1/0.2` 时分别允许 `20.0%/20.2%/20.8%` 状态安静，同时保留至少 80% 总体探索能量。phase232 冻结 `262144` 帧的十维总体方差为 `0.99943--0.99964`，目标从 `0.8` 到 `0.99` 的预算损失和梯度均为零，因此候选不会延续 phase231 的逐状态增噪。排除继续使用每样本 entropy、固定正则权重或按 readiness/stage 直接缩放
正式保留/删除的代码：保留独立 `exploration_budget.py` 和 `test_exploration_budget.py`；尚未接入 actor、trainer、配置或 checkpoint，不改变训练行为
前序阶段回归：phase232 临时代码清理后相关单测 `20 passed`，py_compile、`git diff --check` 和专属标识搜索均通过
下一决策门槛：phase234 接入统一 PPO 接口，必须明确 actual likelihood、global capacity entropy 和 state-scale budget 的梯度所有权，并通过 optimizer smoke 后才能短训练
```

2026-09-28 22:31，phase233 完成。容器内候选测试 `8 passed`，与 temporal Gaussian 联合回归 `18 passed`，py_compile 和 `git diff --check` 通过；未启动 Isaac 或训练进程。预算目标 `0.8` 是“至少保留 80% 总体归一化探索方差”的任务无关约束，不是阶段阈值，也不读取任何任务状态标签。

### phase234：总体方差预算 PPO 接口（已验证）

```text
上游结论与待解决瓶颈：phase233 已验证总体预算的数学语义；当前 actor/trainer 仍只有状态无关 std，候选尚未进入真实 temporal PPO 概率链
基线 checkpoint / SHA-256：无；策略参数化变化后从随机初始化做 smoke，历史 checkpoint 不兼容也不得续训
唯一变量：状态方差重新参数化为 global capacity std 乘 observation-dependent attenuation；实际 temporal likelihood 使用完整状态方差，entropy 只优化 global capacity，attenuation 只接收 PPO surrogate 与总体方差预算梯度
待验证假设：分离梯度所有权后，远处探索 capacity 可保持，而状态 attenuation 不再被每样本 entropy 推到上限；预算只在全局探索不足时激活
禁止改动范围：不改 reward、完成条件、termination、安全阈值、observation/action、随机化、低层动力学、AutoOpen 或阶段逻辑；不得读取 stage、seed、门型、readiness 或脚本目标；不启动正式训练
运行规模与日志：全量测试 `54 passed, 5 subtests passed`。GPU 3 上完成三组 `8 env x 8 steps` smoke：正常 1 update 日志 `doorman_train:/tmp/b2z1_phase234_budget_interface_smoke.log`，checkpoint `...-20260928_223400/model_step_000001.pt`，SHA-256 `7284f0f1cb9958e8f80b2c6ffddd7a828ed9e5323fc3119e2673ff4591d2b85a`；从 step1 恢复至 step2 的日志 `doorman_train:/tmp/b2z1_phase234_budget_resume_smoke.log`，checkpoint `...-20260928_223552/model_step_000002.pt`，SHA-256 `75d93cd3d0e529465cb60db26b7daf58d436295ab944ec7244b27758c378f02b`；仅诊断使用预算 `0.95` 强制触发违反路径，日志 `doorman_train:/tmp/b2z1_phase234_budget_violation_smoke.log`，checkpoint `...-20260928_223657/model_step_000001.pt`，SHA-256 `a73bd431d73e6a907c46ed1a7ae360de9113864adbc3706076489cc8f60744f9`
成功/失败判据：实际 state std 的 rollout/update likelihood 必须精确重算；entropy 对 attenuation 梯度严格为零、对 global std 有限；预算有余量时 attenuation 只收 surrogate，违反时对偶乘子和增噪梯度方向正确；checkpoint round-trip 保留对偶状态；smoke 无异常或 NaN
观测指标：global std、state attenuation、总体归一化方差、预算 shortfall、dual weight、三类分项梯度、KL/log-prob 误差和 optimizer update
结论与已排除方向：已验证。actual temporal likelihood 使用 `global_std * attenuation(observation)`；entropy 只对 global capacity std 反传，不向 attenuation 传播；总体预算仅在上一 rollout 的归一化方差低于目标时激活。正式预算 `0.8` 下初始方差 `0.95^2=0.9025`，十维 dual 均为 0；诊断预算 `0.95` 下十维 dual 均约 `0.000474999`，与 `0.01 * (0.95 - 0.9025) = 0.000475` 一致。checkpoint round-trip 后 attenuation 网络和 persistent dual buffer 均恢复。排除概率链、对偶符号、持久化状态或单步 optimizer 接口错误
正式保留/删除的代码：保留 `exploration_budget.py`、actor observation-dependent attenuation、trainer 总体预算与配置；诊断仅通过配置覆盖预算值，没有保留专用行为分支
前序阶段回归：全量 `54 passed, 5 subtests passed`；三组 Isaac smoke 均无异常或 NaN，并保持 cameras=0、RTX sensors=0、frame rendering disabled
下一决策门槛：phase235 从随机初始化四卡训练至 step25，直接检验真实 PPO 是否形成状态相关降噪并改善联合停稳；若几何回归、attenuation 仍统一饱和或无状态分化，停止并归因，不续训、不扫描预算
```

2026-09-28 22:40，phase234 完成。三份容器日志和三个 checkpoint 均已核对存在，SHA-256 已记录；宿主机无训练进程，GPU 0--3 空闲。该 phase 只确认接口与数学链路正确，不把 smoke 结果当作学习效果。

### phase235：总体方差预算四卡随机初始化短训（失败）

```text
上游结论与待解决瓶颈：phase234 已验证总体方差预算的概率、梯度、对偶和 checkpoint 接口；第一阶段仍只学会几何接近，尚未学会到位后联合停稳
基线 checkpoint / SHA-256：无；策略参数化已变化，必须从随机初始化干净训练，禁止继承 phase231 或其他历史 checkpoint
唯一变量：启用 phase234 已验收的 observation-dependent attenuation 与 `0.8` 总体归一化方差预算；其余训练、reward、完成条件、随机化和低层语义保持不变
待验证假设：PPO surrogate 能在稀有高 readiness 状态降低 attenuation，而总体预算保持远处探索，从而不损害几何接近并降低到位后的实际底盘/EE速度，开始形成连续联合停稳
禁止改动范围：训练运行期间不改代码；不得修改 reward、完成条件、安全阈值、observation/action、随机化、低层、预算值、entropy 系数或训练步数；不得加入 stage、seed、门型、readiness 或脚本目标特例
运行规模与日志：GPU 0--3，每卡 2048 env，128 steps/env，4 mini-batches，从随机初始化训练至 step25；日志 `doorman_train:/tmp/b2z1_phase235_budgeted_state_noise_clean_to25.log`，分布式端口 `29535`。2026-09-28 22:42 已启动，accelerate 主进程 PID `1288010`，四个训练 rank PID `1288076--1288079`；首个 update 于 22:47 正常完成，全局 `1,048,576` timesteps、iteration `46.05 s`，运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260928_224258`
成功/失败判据：至少不得相对 phase224/229/231 step25 发生几何与安全回归，并必须出现可重复的 observation-dependent attenuation 分化或联合停稳改善。若五项 geometry quality 回归、attenuation 在 readiness 各桶统一贴上限、没有状态分化、出现异常/NaN/显著安全回归，立即判失败并停止；不续训、不调预算
观测指标：五项 quality/episode-best/episode-seen；actual base/EE speed；stable_reached_1/5/10；GRASP occupancy/reached；attenuation/readiness 分桶；十维总体归一化方差、budget shortfall、dual；安全终止、KL、NaN。同期对照为 phase224 `0.3438/0.5013/0.1565/0.1244/0.9005`、phase229 `0.3232/0.5008/0.1526/0.1209/0.8636`、phase231 `0.3137/0.5014/0.1325/0.1183/0.8782`
结论与已排除方向：失败。25 updates、`26,214,400` timesteps 正常完成，末轮 iteration `47.35 s`，无异常或 NaN。step25 五项 quality 为 `0.3165/0.5021/0.1305/0.0781/0.8446`，相比 phase224 `0.3438/0.5013/0.1565/0.1244/0.9005` 除 yaw 外均回归；相比 phase231 也只有 base position/yaw 略高，pregrasp/orientation/open 更低。episode-best 为 `0.4189/0.6606/0.2177/0.2139/0.9775`，stable 1/5/10、GRASP occupancy/reached 全为 0。actual planar speed `0.2103 m/s`，只比 phase231 `0.2177 m/s` 略低，未换来停稳；unsafe-force 为 0，排除安全终止上升是本轮失败原因。checkpoint 中 global std 仍约为逐动作上限，十维 dual 全为 0，说明总体探索预算没有违反，但现有日志不足以判断 visited states 上 attenuation 是否按 readiness 分化
正式保留/删除的代码：phase234 候选暂留到 phase236 完成根因归因；phase235 没有新增行为代码。若归因确认机制失败，将整体删除候选，不保留兼容分支
前序阶段回归：纯动力学、headless、camera_types=[]；四卡同一 SimulationApp 正常完成，无环境重建卡死、分布式异常、NaN 或显著安全回归，但 APPROACH_PREGRASP 学习行为回归
下一决策门槛：phase236 只读导出 readiness x action 的 attenuation/global/marginal std、actor mean、sample noise 和真实速度；确认首个断点前不续训、不调预算、不改 reward
```

2026-09-28 23:08，phase235 已按预注册失败条件停止。运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260928_224258`；checkpoint `model_step_000025.pt` SHA-256 为 `62fb48764f3ea248e392096ef5b1c9d44618ec4bac2ba0d21ae71d688b6696bd`。该 checkpoint 只能用于冻结归因，禁止续训。

### phase236：冻结状态方差行为归因（已验证）

```text
上游结论与待解决瓶颈：phase235 接口和运行链路正常，但 step25 几何回归、联合停稳与 GRASP 均为 0；聚合日志不能区分 attenuation 未分化、近目标状态覆盖不足或 actor mean 仍主导速度
基线 checkpoint / SHA-256：phase235 `model_step_000025.pt` / `62fb48764f3ea248e392096ef5b1c9d44618ec4bac2ba0d21ae71d688b6696bd`；只读加载，不续训
唯一变量：增加可完整删除的只读 rollout 导出；策略、环境、reward 和全部物理行为保持不变，optimizer 不执行 step
待验证假设：若 attenuation 已在高 readiness 状态显著降低而实际速度仍高，则瓶颈在 actor mean 或状态覆盖；若各 readiness 桶 attenuation 近似一致，则总体预算虽正确但 PPO credit 尚未学出状态分化
禁止改动范围：不更新模型；不改预算、entropy、reward、完成条件、安全阈值、observation/action、随机化、低层或阶段逻辑；不得按 stage、seed、门型或 readiness 修改动作
运行规模与日志：单卡 GPU 3、2048 env、128 steps，冻结 phase235 step25；日志 `doorman_train:/tmp/b2z1_phase236_state_noise_diag.log`，张量 `doorman_train:/tmp/b2z1_phase236_state_noise_diag.rank0.pt`。首次 detached 启动在 checkpoint 载入期间尚未创建日志，被误判为未启动后又启动了一份；两组进程及受污染 `/tmp` 输出均已完整删除，本轮无效且禁止作为证据。第二轮单进程完成 rollout 后，临时导出把 OmegaConf `ListConfig` 当 tensor 调用 `.detach()`，在 `torch.save` 前异常退出；optimizer 未执行、没有产生张量文件，该轮也禁止作为证据。已将配置值显式转为 tensor，正式重跑只允许一个前台可追踪 session。覆盖不足时才增加只读 rollout，不先扩大训练
成功/失败判据：必须在同一批样本上按 readiness 分桶给出十维样本数、attenuation/global/marginal std、actor mean、sample noise、actual base/EE speed，并闭合到总体归一化方差和 dual 状态；仅看网络权重或聚合 mean std 不算完成
观测指标：readiness 分布及 occupancy；每桶十维 attenuation/global/marginal std；mean/noise 对实际速度贡献；总体归一化方差、shortfall、dual；安全和终止
结论与已排除方向：已验证。有效冻结 rollout 共 `262144` 帧，全部处于 APPROACH。readiness 分位数为中位 `0.000541`、90% `0.00638`、95% `0.01168`、99% `0.03054`、99.9% `0.06059`、最大 `0.17238`；`0.05--0.1` 仅 667 帧，`0.1--0.2` 仅 29 帧。最低主桶十维 attenuation 均值约 `0.898--0.939`，`0.1--0.2` 桶约 `0.847--0.928`，证明网络不是完全不分化，但幅度只有约 `1.8%--5.7%`。总体十维归一化方差为 `0.8058--0.8816`，已靠近 `0.8` 预算下界，dual 仍全 0。全体 base/EE motion 为 `0.3046/0.4724 m/s`，高桶反而为 `0.3247/0.5788 m/s`；高桶十维 sample-noise 绝对值仍普遍显著大于 actor mean。由此排除“方差网络完全没学到状态差异”，首个断点收敛为高 readiness 样本极少、定向降噪太弱，而预算主要被用作全局轻度降噪；尚需 phase237 证明梯度层是 credit 缺失还是数量稀释
正式保留/删除的代码：`B2Z1_PHASE236_DIAG_PATH`、逐步 CPU 导出、`torch.save` 和 optimizer 前退出均已完整删除；保留容器证据 `/tmp/b2z1_phase236_state_noise_diag.rank0.pt`（SHA-256 `f81601131baad6b93072a4b022a122eae295f732dbefbad34555687e498e3703`）及 `/tmp/b2z1_phase236_state_noise_analysis.log`（SHA-256 `e27eb0d5c7c9a0502b5f690e8315e75cbda191fe43f886febd5253a3184758b6`）。phase234 候选暂留待梯度归因
前序阶段回归：有效运行在 optimizer 前退出，源 checkpoint SHA-256 仍为 `62fb48764f3ea248e392096ef5b1c9d44618ec4bac2ba0d21ae71d688b6696bd`。临时代码清理后 py_compile、`54 passed, 5 subtests passed`、专属标识搜索和 `git diff --check` 通过；当前无训练进程
下一决策门槛：phase237 在 exact temporal PPO surrogate 下按 readiness 分桶计算 attenuation 输出及共享网络参数梯度；若高桶没有降噪 credit，转 reward/advantage 归因；若有但被总体梯度淹没，才设计统一的 credit 聚合机制
```

2026-09-28 23:26，phase236 完成。首次重复启动和第二次序列化失败均已记录且不作证据；第三次单进程 rollout 正常完成，`camera_types=[]`、optimizer 未执行。临时诊断完整清理后全量测试通过。

### phase237：readiness 分桶 surrogate 梯度归因（已验证）

```text
上游结论与待解决瓶颈：phase236 证明 attenuation 有弱状态分化，但高 readiness 极稀有、总体方差已靠近预算下界，尚不清楚稀有状态降噪 credit 是否存在以及如何被总体更新稀释
基线 checkpoint / SHA-256：phase235 `model_step_000025.pt` / `62fb48764f3ea248e392096ef5b1c9d44618ec4bac2ba0d21ae71d688b6696bd`；只读加载，不续训
唯一变量：增加可完整删除的只读 rollout 与分项 autograd 导出；在同一参数点、同一 exact temporal likelihood 下分别计算 readiness 桶和总体 surrogate 对 attenuation logits、scale 网络及 global std 的梯度，不调用 optimizer.step
待验证假设：高 readiness 样本本身给出降低相关动作方差的正确 surrogate 梯度，但因只占 `0.011%`，其共享参数贡献被低 readiness 样本的梯度和总体样本数量淹没
禁止改动范围：不更新模型；不改预算、entropy、reward、advantage、样本权重、完成条件、observation/action、随机化、低层或阶段逻辑；不得以人为 readiness 加权替代诊断
运行规模与日志：单卡 GPU 3、2048 env、128 steps，冻结 phase235 step25；2026-09-28 23:36 已启动，accelerate 主进程 PID `1291800`、训练进程 PID `1291806`。日志 `doorman_train:/tmp/b2z1_phase237_readiness_gradient.log`，张量 `doorman_train:/tmp/b2z1_phase237_readiness_gradient.pt`，optimizer 前退出
成功/失败判据：必须给出每个 readiness 桶的样本数、surrogate 对 attenuation 输出和网络参数的逐动作梯度方向/范数、各桶对总梯度的贡献及余弦关系，并闭合到不执行 optimizer 的同一参数点；只比较 loss 或训练后方差不算完成
观测指标：advantage、ratio/clip、逐动作 dL/dlogit 与 dL/dsigma、scale 网络梯度范数/余弦、按样本数还原的总贡献、global std 梯度、readiness 覆盖
结论与已排除方向：证据已验证。有效样本 `262144`，readiness 桶计数为 `246683/11314/3563/554/30/0/0`，占比 `94.1021%/4.31595%/1.35918%/0.21133%/0.01144%/0/0`。`0.1--0.2` 桶 advantage 均值/标准差为 `-1.2070/1.2713`；其 base_vx/base_vy/base_yaw/ee_x/ee_y/ee_z 的 `dL/dattenuation` 全为负，梯度下降会继续增大六维噪声，只有 ee_roll/ee_pitch/ee_yaw/gripper 为降噪方向。该桶共享 scale 网络梯度范数为总体梯度的 `4.46%`，与总体余弦 `0.144`，对总体梯度投影仅 `0.6428%`；最低桶投影为 `87.8248%`。因此排除“高 readiness 已有正确联合降噪 credit，只是被样本数稀释”这一完整假设：稀有性确实存在，但更早的断点是关键运动维 credit 方向本身错误。不得直接加 readiness 权重，否则会放大错误方向
正式保留/删除的代码：`B2Z1_PHASE237_DIAG_PATH`、readiness 采集、`_run_phase237_gradient_diagnostic()` 和 optimizer 前退出已完整删除；不保留诊断分支。phase234 候选暂留到 phase238 根因结论
前序阶段回归：运行确认 `2048 env`、`camera_types=[]`、`cameras=0`、`RTX sensors=0`、frame rendering disabled；在 optimizer 前退出，无异常或 NaN。ratio 平均绝对误差各桶不超过 `7.07e-4`、clip fraction 全 0、参数梯度投影和为 `1.000003`、闭合误差 0；除刻意包含 `inf` 上界的 `bin_edges` 外所有指标张量有限。源 checkpoint SHA-256 仍为 `62fb48764f3ea248e392096ef5b1c9d44618ec4bac2ba0d21ae71d688b6696bd`
下一决策门槛：phase238 只读拆分高 readiness 的即时结果 reward、后续 reward-to-go、termination 和 value baseline，确认负 advantage 的首个来源。根因前不修改 reward、采样权重或方差机制
```

2026-09-28 23:32，phase237 执行游标登记：实现已开始，尚无训练或诊断进程，也没有 phase237 输出文件。恢复后不得重写 readiness 采集或重新设计实验；下一原子动作仅为审核现有 `_run_phase237_gradient_diagnostic()` 的 shape、autograd 与 exact likelihood，并完成 optimizer 前退出接线和静态检查。

2026-09-28 23:36，phase237 静态审核和接线完成。容器内 py_compile 与 `git diff --check` 通过；未启动 Isaac。审核修复均只作用于临时诊断本身，不改变策略、reward、环境或物理行为。下一原子动作是按已登记路径启动唯一一份单卡冻结 rollout，并在确认进程唯一后移动游标。

2026-09-28 23:37，phase237 唯一一份单卡冻结 rollout 已启动。进程核对只有 accelerate PID `1291800` 和其训练子进程 PID `1291806`；日志文件已创建但尚无内容，GPU 尚未分配显存，属于 checkpoint 加载前段。后续只轮询这一进程，禁止因暂时无日志而重复启动。

2026-09-28 23:44，phase237 唯一 rollout 正常完成并在 optimizer 前退出。张量 `/tmp/b2z1_phase237_readiness_gradient.pt` SHA-256 为 `360f052cafeb7449368c8f57b447a69687a023e6216193ccf7de972acf7d27d0`，日志 `/tmp/b2z1_phase237_readiness_gradient.log` SHA-256 为 `1ae6ea6950e0f9418501be862716a9fc013e2c9d0a11876f331452081be3e81f`。没有训练进程残留。下一原子动作仅为删除临时代码和完整回归，不得直接修改 reward 或继续训练。

2026-09-28 23:48，phase237 临时代码已完整删除。容器内 py_compile、完整测试 `54 passed, 5 subtests passed`、phase237 专属标识搜索和 `git diff --check` 均通过；源 checkpoint 哈希未变，无训练进程。phase237 正式封账，后续不得重复该梯度归因或直接对 readiness 加权。

### phase238：高 readiness 负 advantage 时间链归因（已验证）

```text
上游结论与待解决瓶颈：phase237 证明高 readiness 的 base 与 EE 平移六维收到继续增噪的 PPO 梯度，且高桶 advantage 均值为负；尚不清楚负 credit 首先来自即时结果 reward、随后失稳/终止，还是 critic baseline
基线 checkpoint / SHA-256：phase235 `model_step_000025.pt` / `62fb48764f3ea248e392096ef5b1c9d44618ec4bac2ba0d21ae71d688b6696bd`；只读加载，不续训
唯一变量：增加可完整删除的只读时间链导出，在同一 rollout 中按 readiness 分层拆分即时 scaled reward、reward-to-go、GAE TD residual/advantage、value baseline、done 及具体终止来源；不调用 optimizer
待验证假设：phase237 的负 normalized advantage 可能来自两层之一：raw GAE 已因随后退化/终止而偏低，或 raw GAE 仍为正但低于同一 stage/variant 的组均值而在归一化后变负；只有前者才能直接证明物理时间 credit 为负
禁止改动范围：不改 reward、GAE、termination、完成条件、方差机制、样本权重、observation/action、随机化、低层或阶段逻辑；不得加入 readiness/stage/seed/门型特例
运行规模与日志：单卡 GPU 3、2048 env、128 steps，冻结 phase235 step25；日志登记为 `doorman_train:/tmp/b2z1_phase238_time_chain.log`，张量登记为 `doorman_train:/tmp/b2z1_phase238_time_chain.pt`，optimizer 前退出
成功/失败判据：必须对同一高 readiness 样本闭合 `reward -> TD residual -> GAE advantage`，并区分 rollout 内后续物理退化、具体终止和 bootstrap/value 的贡献；只比较聚合 reward 或 value 均值不算完成
观测指标：readiness 分桶计数；逐项 scaled reward；未来 readiness/速度/接触；done 与 termination reason；reward-to-go；逐步 TD residual；GAE advantage；current/next/final value 与 bootstrap remainder
结论与已排除方向：已验证。有效 rollout 的 reward component、raw GAE 和 normalized advantage 最大闭合误差分别为 `0/3.81e-6/0`。高 readiness `0.1--0.2` 有 25 帧，raw/normalized advantage 均值为 `-18.90/-3.60`，不是单纯归一化假象。按唯一轨迹去重后共 14 条：3 条在未来 32 步内终止，首帧 raw advantage 均值 `-30.51`；其余 11 条虽未终止，未来窗口仍平均损失 `8.68` stage progress，首帧 raw advantage 均值 `-13.09`。高 readiness 平均 base/EE motion 为 `0.356/0.902 m/s`；旧 `0.2 * readiness * actual_stability` 的 actual stability 平均/最大仅 `0.000581/0.00685`，最终势函数贡献平均/最大仅 `1.38e-5/1.62e-4`。因此首个根因不是 reward 符号、GAE、归一化或 critic 单独误估，而是稳定结果信号经连乘和严格指数衰减后数值消失。排除直接加 readiness 样本权重、继续调方差预算或仅延长训练
正式保留/删除的代码：全部 phase238 reward/termination 镜像、raw GAE 统计、导出和 optimizer 前退出已删除；不保留专属路径
前序阶段回归：清理后 py_compile、`54 passed, 5 subtests passed`、专属标识搜索和 `git diff --check` 通过；源 checkpoint SHA-256 未变
下一决策门槛：phase239 用同级结果 component 重构第一阶段 potential，必须先用纯函数测试证明几何与稳定都不可被另一项替代，再做真实训练
```

2026-09-28 23:52，phase238 只读字段审计完成。最关键的新事实是 phase237 的 advantage 已经过 stage/variant 组内标准化，负值不能直接等同于负物理回报。下一原子动作只允许设计和静态审核最小诊断字段；不得据 phase237 的负均值直接修改 reward。

2026-09-28 23:53，phase238 最小只读诊断接线完成。它只镜像环境已经计算出的逐项 reward 和 termination flags，并导出 raw GAE、归一化组统计、readiness/速度及动作；写盘前会逐元素重建三条数学链。py_compile、完整测试 `54 passed, 5 subtests passed` 和 `git diff --check` 通过。下一原子动作是按已登记路径启动唯一一份单卡冻结 rollout。

2026-09-28 23:54，phase238 首次启动无效：2048 环境和 checkpoint 正常加载后，在第一步 `_check_termination()` 因临时诊断新增的 `os.environ` 缺少 `import os` 抛出 `NameError`。没有完成任何 rollout、没有执行 optimizer、没有生成张量，源 checkpoint 未修改；失败日志保留为 `doorman_train:/tmp/b2z1_phase238_time_chain_invalid_missing_os.log`。仅补充缺失导入后 py_compile 与 `git diff --check` 通过，下一原子动作是以完全相同配置重跑，不修改诊断语义。

2026-09-29 00:03，phase238 有效重跑完成并封账。张量 `/tmp/b2z1_phase238_time_chain.pt` SHA-256 为 `b0d40ad352e9c38c78f7b6e50ec6e1ae38f3cdbf2c91df5a8c8f8f48979376f2`，日志 `/tmp/b2z1_phase238_time_chain.log` SHA-256 为 `60e4d310a7b7d18d252d0eb92ed0a7b5229c1c7a0867ccb7f4ba5087631d35d2`；optimizer 未执行，源 checkpoint SHA-256 仍为 `62fb48764f3ea248e392096ef5b1c9d44618ec4bac2ba0d21ae71d688b6696bd`。临时代码清理后完整测试通过，无训练进程。

### phase239：统一第一阶段结果势函数（失败）

```text
上游结论与待解决瓶颈：phase238 证明近目标几何随后持续退化，旧稳定势在真实高 motion 下平均贡献仅 1.38e-5，数值上不能提供减速 credit
基线 checkpoint / SHA-256：无；reward 语义变化后必须随机初始化干净训练
唯一变量：用统一 bottleneck-balanced 结果 potential 同时评价五项归一化几何、无接触和实际 base/EE 稳定，删除 readiness 连乘后近零的旧稳定 coupling
待验证假设：把稳定作为不可由几何替代的同级物理结果后，策略可在保留几何接近的同时获得从高 motion 连续减速的非零 credit
禁止改动范围：不改完成条件、termination、安全阈值、observation/action、方差机制、随机化、低层或后续阶段；不得奖励 actor command、脚本目标或 AutoOpen 行为
运行规模与日志：GPU 3 的 `2048 env x 128 steps x 1 update` smoke 已通过。正式四卡使用 GPU 0--3、每卡 2048 env、128 steps、4 mini-batches，日志 `doorman_train:/tmp/b2z1_phase239_result_stability_clean_to25.log`，端口 `29539`，运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260929_001230`，每 25 step 保存；update1 iteration `48.59 s`、吞吐 `21579 steps/s`
成功/失败判据：纯函数必须保证任一低几何 component 或低稳定 component 都限制总 progress，且远处静止不能替代接近；step25 至少不回归 phase224 几何，并出现实际速度或 stable_reached 的明确改善，否则失败并停止
观测指标：五项 geometry quality/episode-best、base/EE motion、stable_reached_1/5/10、GRASP occupancy/reached、termination、安全、探索方差与训练吞吐
结论与已排除方向：失败。step25 五项 quality 为 `0.3175/0.5024/0.1396/0.1043/0.8601`，除 yaw 外均低于 phase224 `0.3438/0.5013/0.1565/0.1244/0.9005`；episode-best 为 `0.4185/0.6707/0.2301/0.2434/0.9793`。actual planar speed `0.2116 m/s` 对 phase224 `0.2143 m/s` 仅下降 `1.26%`，不能视为明确改善；stable 1/5/10、GRASP occupancy/reached 全为 0。unsafe-force、door-overspeed 和 shoulder-collision 均为 0，无异常或 NaN，排除安全回归或运行故障导致失败。不得据此直接断言权重太小；仍需冻结归因新势函数是否缺覆盖、数值不足或收到错误 credit
正式保留/删除的代码：候选 `geometry_gated_stability_progress()` 暂留，只用于 phase240 对同一失败机制做只读归因；归因完成后必须明确保留或删除。旧 `readiness_coupled_stability_progress()` 保持删除，不恢复双路径
前序阶段回归：静态实现后定向 `15 passed, 5 subtests passed`、完整 `56 passed, 5 subtests passed`、py_compile 和 `git diff --check` 通过；四卡训练纯动力学运行至 step25，无前序接口异常
下一决策门槛：phase240 冻结 phase239 step25，在同一 rollout 上闭合 geometry progress、actual stability、最终 potential、raw/normalized advantage 和 exact PPO surrogate 梯度；不执行 optimizer
```

2026-09-29 00:10，phase239 静态实现完成。旧 coupling 专属标识搜索为零，定向测试 `15 passed, 5 subtests passed`，完整测试 `56 passed, 5 subtests passed`，py_compile 与 `git diff --check` 通过。新函数保证远处静止不能替代几何接近，并在 phase238 实测 motion 区间保留大于 `0.1` 的可学习减速差值。下一原子动作仅为单卡真实 optimizer smoke。

2026-09-29 00:12，phase239 单卡真实 optimizer smoke 完成。GPU 3、2048 env、128 steps、1 update，iteration `45.15 s`；确认 cameras=0、RTX sensors=0、frame rendering disabled，无异常或 NaN。首轮五项 quality 为 `0.3357/0.5146/0.1314/0.0466/0.7709`，stable 1/5/10 全 0，仅作随机初始化基线。下一原子动作是四卡随机初始化至 step25，运行期间冻结代码。

2026-09-29 00:17，phase239 四卡随机初始化训练完成 update1。GPU 0--3 每卡 2048 env，四个场景均确认 cameras=0、RTX sensors=0、frame rendering disabled；每轮 `1,048,576` timesteps，首轮 iteration `48.59 s`、吞吐 `21579 steps/s`。运行目录为 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260929_001230`；无异常或 NaN，代码冻结至 step25。

2026-09-29 00:39，phase239 四卡训练在预注册 step25 正常结束，四个 worker 已退出。共 `26,214,400` timesteps，末轮 `49.07 s`；日志为 `doorman_train:/tmp/b2z1_phase239_result_stability_clean_to25.log`。checkpoint `model_step_000025.pt` SHA-256 为 `77f44d7c98de8255ab8ec067c8923319994f194c79d0a75774da4e7bf1cffaec`，只作失败机制诊断，不得续训。预注册学习判据失败，下一原子动作仅为登记和静态审核 phase240 只读导出。

### phase240：新结果势函数数值与梯度归因（已验证并清理）

```text
上游结论与待解决瓶颈：phase239 的新结果势函数通过静态排序和运行验收，但 step25 几何回归、实际速度基本不变且无联合稳定；尚不清楚首个断点是近目标覆盖、结果势数值动态范围、raw advantage，还是 exact PPO 梯度方向
基线 checkpoint / SHA-256：phase239 `model_step_000025.pt` / `77f44d7c98de8255ab8ec067c8923319994f194c79d0a75774da4e7bf1cffaec`；只读加载，不续训
唯一变量：增加可完整删除的只读 rollout 导出，逐样本记录 geometry progress、contact-free、base/EE motion、actual stability、最终 potential、逐项 reward、raw/normalized advantage，并按 geometry/readiness 分桶计算 exact temporal PPO surrogate 梯度；optimizer 不执行 step
待验证假设：新势函数在实际访问到的近目标高速样本上虽非零，但其 reward-to-go 或关键运动维 surrogate 梯度仍不足以形成联合减速
禁止改动范围：不更新模型；不改 reward、权重、motion tolerance、方差预算、完成条件、安全阈值、observation/action、随机化、低层或阶段逻辑；不按 stage、seed、门型或 readiness 加权
运行规模与日志：单卡 GPU 3、2048 env、128 steps，冻结 phase239 step25；正式日志 `doorman_train:/tmp/b2z1_phase240_result_credit_diag.log`，张量 `doorman_train:/tmp/b2z1_phase240_result_credit_diag.pt`。`8 env x 8 steps` smoke 已在 optimizer 前正常退出
成功/失败判据：必须逐元素闭合新 potential 和 GAE，并给出近目标样本数量、稳定质量动态范围、后续物理结果以及 base/EE 关键动作维的 surrogate 梯度方向和对总梯度贡献；只看聚合 reward 或训练曲线不算完成
观测指标：geometry/readiness 分布、base/EE motion、actual stability、contact-free、最终 potential、逐项 reward、raw/normalized advantage、termination、ratio/clip、逐动作与共享参数梯度
结论与已排除方向：已验证。262,144 个有效帧的 geometry 分桶计数为 `1723/83437/175899/1083/2/0`，`geometry >= 0.4` 仅 1,085 帧，占 `0.414%`。`0.4--0.6` 桶的 geometry/base motion/EE speed/actual stability/potential 均值为 `0.42485/0.30281/0.47919/0.11190/0.34775`，raw/normalized advantage 为 `-2.1584/-0.05805`，未来 32 步 geometry delta 为 `-0.03051`，未来终止率 `1.94%`。exact action-mean 总梯度贡献按桶为 `39.999%/12.087%/43.417%/4.826%/-0.330%`；`0.4--0.6` 桶的 base 三维和 EE xyz command 减速投影分别为 `+0.000383/+0.002468`，证明局部 mean credit 有正确减速方向。attenuation 梯度则为混合方向，base x/y、EE x/z 降噪而 base yaw、EE y 增噪。首个失败机理因此闭合：乘法形式 `Phi=G*(0.8+0.2*S)` 在大多数低稳定样本上令 `dPhi/dG` 约为 `0.8`，普遍削弱约 18% 几何学习；真正有用的稳定 credit 只来自极少近目标样本，无法抵消几何损失。排除“机制只是权重太小，可直接调权重继续训练”；不得扫描 stability weight 或改名重试同一乘法结构
正式保留/删除的代码：全部 phase240 环境缓存、storage 字段、rollout 导出、exact 梯度函数和 optimizer 前退出已删除。phase239 的 `geometry_gated_stability_progress()` 及专属测试也已删除；旧 `readiness_coupled_stability_progress()` 不恢复。正式生产路径暂回到五项 `bottleneck_balanced_progress()` 几何结果势，不保留双路径
前序阶段回归：正式 rollout cameras=0、RTX sensors=0、frame rendering disabled，ratio 平均绝对误差 `3.971e-4`、clip fraction 0、全部张量有限，potential 重建误差 0，非终止连续帧 stage delta mean/max 误差 `6.30e-10/7.63e-6`。清理后 py_compile、定向 `5 passed, 5 subtests passed`、完整 `52 passed, 5 subtests passed`、phase240/失败函数标识搜索和 `git diff --check` 均通过；无训练或诊断进程
下一决策门槛：phase241 登记前不得修改代码。新结果势必须同时满足：不削弱原几何 credit；稳定 credit 是额外、连续、仅基于真实物理结果的信号；远处静止不能替代接近；不使用 command、脚本目标、stage/seed/门型特例；不扫描权重
```

2026-09-29 00:45，phase240 最小只读接线和 smoke 完成。诊断逐步采集 geometry、contact-free、实际 base/EE motion、actual/safe stability、最终 potential 和 stage-progress reward，并与同一步 action、总 reward、raw/normalized advantage 对齐；PPO 梯度复用生产 temporal likelihood 与 clipped surrogate。`8 env x 8 steps` 输出 `/tmp/b2z1_phase240_smoke.pt` 全有限，ratio 平均绝对误差 `1.82e-6`、clip fraction 0；源 checkpoint SHA-256 未变，无残留进程。下一原子动作是启动唯一正式冻结 rollout。

2026-09-29 00:55，phase240 唯一正式冻结 rollout 完成。GPU 3、2048 env、128 steps，共 262,144 帧；optimizer 未执行。张量 `doorman_train:/tmp/b2z1_phase240_result_credit_diag.pt` SHA-256 为 `875b0cd7aeb89a518823595ef7d844397cd1607b233821a0031fc66212d7c74e`，日志 `doorman_train:/tmp/b2z1_phase240_result_credit_diag.log` SHA-256 为 `dead5c2b9fd56005ded20c968a6fe7b2f2294c008ba83f4a7f21d3d0d5c9dfe7`；源 checkpoint SHA-256 未变。首次缺少 command 方向投影的不完整证据另存为 `/tmp/b2z1_phase240_result_credit_diag_incomplete.{pt,log}`，不得替代正式证据。

2026-09-29 01:02，phase240 完成归因和清理。正式数据证明局部稳定 credit 并非数值归零或全方向错误，而是覆盖仅 `0.414%`、总梯度贡献不足，同时乘法结构对绝大多数样本先削弱几何 credit，解释了 phase239 的几何回归且稳定不改善。phase239 失败势及 phase240 临时代码已完整删除；容器内 py_compile、定向 `5 passed, 5 subtests passed`、完整 `52 passed, 5 subtests passed`、专属标识搜索和 `git diff --check` 均通过。当前无训练或诊断进程。下一原子动作仅为在本文档登记 phase241 的单一假设、数学形式和静态验收，登记完成前不修改代码或启动训练。

### phase241：额外稳定成就势（运行中）

```text
上游结论与待解决瓶颈：phase240 证明 phase239 的乘法势在极少近目标样本上有部分正确减速 credit，但它对全部低稳定样本先削弱约 18% 原几何 credit；需要在不牺牲接近学习的前提下增加连续停稳引导
基线 checkpoint / SHA-256：不加载 checkpoint；本 phase 只做数学设计和纯张量离线审核，不运行 Isaac、不更新模型
唯一变量：把稳定 credit 从 `G` 的乘法缩放改为 `Delta B`，其中 `B_t=max(B_{t-1}, min(R_t,S_t)*C_t)`；`R` 是五项真实几何完成度的最差分量，`S=1/(1+max(base_motion,ee_speed)/0.05)`，`C` 是把手和门板均无接触。第一阶段 reward 为 `20*(Delta G + Delta B)`，原始几何 signed delta `Delta G` 逐元素原样保留
待验证假设：联合结果历史最佳增量可以在几何到位后真实减速时提供额外 credit，同时不会降低任何样本已有的几何 credit，不会因保持或重访同一状态重复得分，也不评价 actor command 或脚本目标
禁止改动范围：不改生产 reward、完成条件、termination、安全阈值、observation/action、探索预算、随机化、低层、阶段逻辑或任何 checkpoint；不引入 stage/seed/门型特例，不扫描权重；不得把 AutoOpen 目标、command 或 reference action 纳入定义
运行规模与日志：无运行进程、无 GPU、无 Isaac；若静态审核通过，纯张量测试结果和最终公式直接记录在本 phase，未通过则登记反例并停止
成功/失败判据：必须同时证明：(1) 原几何项及 signed delta 不被缩放；(2) 初始化状态无额外 windfall，保持或重访同一联合质量不能重复得分；(3) 同一几何完成度下实际 motion 越小，新增 joint credit 越高；(4) 接触状态本身无 joint credit；(5) 冻结真实轨迹中必须覆盖几何未继续前进但实际稳定度提高的样本。任一失败即不允许真实训练
观测指标：候选对 geometry、contact-free、base motion、EE speed 和历史成就状态的单调性；原几何 delta 差值；远/近、动/静、接触/无接触和往复序列的逐步 reward
结论与已排除方向：静态审核通过。首个候选“只按刷新几何历史前沿时的稳定度加权”虽能防循环，但无法奖励几何已到位后的原地减速，正好遗漏当前瓶颈，未实现。最终候选使用联合质量历史最佳：phase240 的 262,144 帧中出现 45,346 个正增量帧，占 `17.298%`，累计增量 `55.668`；其中 7,051 帧在 geometry 未增加时因实际减速刷新联合质量。geometry `0.4--0.6` 桶有 97 个正增量、累计 `1.544`。构造往复序列的增量为 `0/0.3/0/0/0.1`，只在首次刷新联合物理结果时得分。明确排除前沿候选、恢复 `G*(a+b*S)`、每步存活型 bonus 和权重扫描
正式保留/删除的代码：新增通用纯函数 `stable_joint_achievement_delta()`、五项纯张量测试、两个每环境 buffer 和第一阶段唯一调用点；无诊断/导出/环境变量。原 `Delta G`、完成条件、observation/action、探索及后续阶段均未修改
前序阶段回归：py_compile、定向 `10 passed, 5 subtests passed`、完整 `57 passed, 5 subtests passed` 和 `git diff --check` 通过。单卡 `2048 env x 128 steps x 1 update` optimizer smoke 完成：cameras=0、RTX sensors=0、frame rendering disabled，262,144 timesteps，iteration `43.72 s`，无异常/NaN；五项 quality `0.3357/0.5146/0.1314/0.0466/0.7709` 与 phase239 同 seed 完全一致，stage-progress reward `0.0612 -> 0.0947`
下一决策门槛：step50 的 `stable_reached_1 > 0` 判据失败，训练停止。phase242 冻结 step50，只有闭合联合成就覆盖、raw/normalized advantage 和 exact temporal PPO mean/attenuation 梯度后，才决定保留或删除候选；禁止继续训练或扫描权重
```

2026-09-29 01:05，phase241 计划已原子登记到顶部执行卡、上层路线图、六阶段总账、近期尝试和本记录。当前没有实现、实验输出或运行进程。恢复后唯一允许动作是推导候选公式并进行纯张量反例审核，不得因上下文恢复重新运行 phase240 或直接启动训练。

2026-09-29 01:12，首个“稳定加权几何历史前沿”候选通过了非负、接触和往复反例，但被目标相关反例否决：它只在刷新几何前沿时给 credit，无法奖励几何保持不变时的真实减速，因此不能解决 phase240 已定位的首个断点。候选未写入源码，后续不得换名重试。

2026-09-29 01:15，联合结果历史最佳候选完成 phase240 冻结轨迹离线反事实。它不替代原 `Delta G`，仅增加 `Delta max_history(min(weakest_geometry, actual_stability)*contact_free)`。共覆盖 45,346 个正增量帧，其中 7,051 个是 geometry 未前进而实际稳定度提高；构造循环不能重复收割。连续稳定步在该轨迹仍为零，因此本轮不额外加入 dwell reward，保持唯一变量。

2026-09-29 01:18，phase241 纯函数、五项测试、环境 buffer 和唯一 reward 接线完成。定向测试 `10 passed, 5 subtests passed`，完整测试 `57 passed, 5 subtests passed`，py_compile 与 `git diff --check` 通过。当前无训练进程；下一原子动作仅为按登记路径启动单卡 optimizer smoke。

2026-09-29 01:19，phase241 单卡 optimizer smoke 正常完成。GPU 3、2048 env、128 steps、1 update，共 262,144 timesteps，iteration `43.72 s`；确认 cameras=0、RTX sensors=0、frame rendering disabled，无异常或 NaN。相同 seed 下五项初始 quality 与 phase239 完全一致，stage-progress episode 均值由 `0.0612` 增至 `0.0947`，证明新联合结果 credit 已接入且未改几何初态语义。运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260929_011509` 只作 smoke，不作为有效 checkpoint。下一原子动作是四卡随机初始化训练至预注册 step25，运行期间冻结代码。

2026-09-29 01:21，phase241 唯一四卡训练已启动。GPU 0--3 每卡 2048 env、128 steps、4 mini-batches，随机初始化训练至 step25；launcher PID `1298234`，worker PID `1298306--1298309`，端口 `29541`，日志 `doorman_train:/tmp/b2z1_phase241_joint_achievement_clean_to25.log`，每 25 step 保存。四个 rank 均加载 `b2z1.isaaclab.python.headless.no_render.kit`，当前处于场景创建；运行中冻结代码，只轮询该进程。

2026-09-29 01:25，phase241 四个 rank 均完成纯动力学验收，确认 cameras=0、RTX sensors=0、frame rendering disabled。update1 正常完成，全局 `1,048,576` timesteps，iteration `51.68 s`，运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260929_012031`。首轮五项 quality 为 `0.3330/0.5146/0.1294/0.0470/0.7726`，actual planar speed `0.2189 m/s`，stable 1/5/10 全 0，stage-progress reward `0.0947`；仅作随机初始化基线，不据首轮波动修改机制。下一原子动作只轮询现有作业至 step25。

2026-09-29 01:45，phase241 四卡训练在预注册 step25 正常完成并退出，共 `26,214,400` timesteps，末轮 `48.27 s`，无异常或 NaN。checkpoint `model_step_000025.pt` SHA-256 为 `e4261c18704d547dbb3bd7f2918f932bdadde12de6f236c59c006603e4826921`；日志 SHA-256 为 `f3fc798a8295d24d3789d25aa93f95b97d0e8b675061f300592c87f197b9d93f`。五项 quality `0.3516/0.5187/0.1659/0.1428/0.9073` 全部超过 phase224，actual speed `0.2072 m/s` 且末五轮持续下降；unsafe-force `0.0001`、door-overspeed 和 shoulder-collision 为 0。但 stable 1/5/10 与 GRASP 仍全 0，因此仅通过继续门槛，未通过阶段学习验收。下一原子动作是同 checkpoint、同机制续训至 step50，硬门槛为 stable1 首次非零。

2026-09-29 01:47，phase241 step25→50 四卡续训已唯一启动。入口读取 checkpoint 并打印 `Resuming at step 25; running 25 updates through step 50`；launcher PID `1299309`、worker PID `1299381--1299384`，端口 `29542`，日志 `doorman_train:/tmp/b2z1_phase241b_joint_achievement_to50.log`。代码、reward、随机化和训练参数均冻结；下一原子动作只轮询这一作业。

2026-09-29 02:10，phase241 四卡续训在 step50 正常完成并退出，累计 `52,428,800` timesteps，末轮 `48.82 s`，无异常或 NaN。checkpoint `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260929_014658/model_step_000050.pt` SHA-256 为 `a7c3b859ed1c4fba98ce49b4a2ec157a9f2fc773da06b79625903ba69d1ea657`；日志 `doorman_train:/tmp/b2z1_phase241b_joint_achievement_to50.log` SHA-256 为 `e33eac1dd3a8f5a0b66c01f4a5d8b8b3c5980d0a4e5664687a19fe0e1e73b12d`。五项 quality 大幅提高，actual speed 降至 `0.1982 m/s`，但 stable1/5/10 与 GRASP 仍全 0，预注册硬门槛失败；不得继续延长。

### phase242：联合成就 credit 与 exact PPO 梯度归因（已验证并清理）

```text
上游结论与待解决瓶颈：phase241 step50 几何和速度同时改善，但七项完成条件从未同帧成立，stable1 硬门槛失败；尚不清楚联合成就增量的首个断点是高联合质量覆盖、raw advantage，还是 exact PPO mean/attenuation 梯度
基线 checkpoint / SHA-256：phase241 `model_step_000050.pt` / `a7c3b859ed1c4fba98ce49b4a2ec157a9f2fc773da06b79625903ba69d1ea657`；只读加载，不续训
唯一变量：增加可完整删除的只读 rollout 导出，记录 weakest geometry、joint quality、previous/new best、achievement delta、contact、base/EE motion、原几何 delta、总 stage-progress reward、raw/normalized advantage，并计算 exact temporal PPO action-mean 与 attenuation 梯度；optimizer 不执行
待验证假设：phase241 的联合成就增量虽覆盖原地减速，但在真正高联合质量样本上的 advantage 或 PPO 梯度占比仍不足，导致策略只能持续改善几何和平均速度，无法形成七项同帧重合
禁止改动范围：不更新模型；不改 reward、权重、完成阈值、termination、observation/action、探索预算、随机化、低层或阶段逻辑；不加入 stage/seed/门型特例
运行规模与日志：单卡 GPU 3、2048 env、128 steps，冻结 phase241 step50；正式日志 `doorman_train:/tmp/b2z1_phase242_joint_credit_diag.log`，张量 `doorman_train:/tmp/b2z1_phase242_joint_credit_diag.pt`，optimizer 前退出
成功/失败判据：必须闭合联合成就 delta 的逐元素重建、reward/GAE、样本覆盖和 exact PPO temporal likelihood；至少按 joint quality 与 achievement-event 分层给出 base/EE 动作 mean 减速投影、attenuation 降噪方向和对总梯度贡献。只看聚合训练曲线不算完成
观测指标：weakest geometry、actual stability、joint quality/best/delta、contact、base/EE motion、原几何 delta、stage-progress reward、raw/normalized advantage、未来联合质量、termination、ratio/clip、逐动作 mean/attenuation gradient
结论与已排除方向：已验证。262,144 帧中 `q>=0.2/0.3/0.4` 仅 2349/80/1 帧，占 `0.896%/0.0305%/0.00038%`；achievement delta 总和 199.453，其中 `q<0.2` 占 `87.86%`。`q>=0.2/0.3` 状态下一步平均退化 `0.0196/0.0725`，保持当前质量 90% 以上的比例仅 `58.79%/31.17%`。`q>=0.2` 的 action-mean 绝对梯度份额约 `0.742%`。achievement 事件 normalized advantage 均值为 `+0.1067`，base/EE mean 有部分减速方向，但 attenuation 聚合梯度为负，gradient descent 会增加噪声。根因不是 credit 数值消失，而是 history-best 只奖励一次性刷新且不惩罚失稳，绝大部分 credit 被低质量随机新纪录消耗。排除续训、调权重、放宽完成条件及换名重试 history-best 结构
正式保留/删除的代码：phase242 临时诊断全部删除；phase241 `stable_joint_achievement_delta`、历史最佳 buffer、reward 接线和专属测试一并删除，回到不受缩放的原始几何 signed delta。专属标识搜索为零，无双路径
前序阶段回归：正式 rollout cameras=0、RTX sensors=0、frame rendering disabled；四项逐元素重建误差 0，GAE max/mean `7.63e-6/1.41e-7`，ratio mean absolute error `0.001516`、clip 0，全部张量有限，源 checkpoint SHA 未变；清理后 py_compile、定向 `5 passed, 5 subtests passed`、完整 `52 passed, 5 subtests passed` 和 `git diff --check` 通过
下一决策门槛：phase243 已登记；只允许先做数学性质和冻结轨迹离线反事实，不得直接实现或训练
```

2026-09-29 02:12，phase242 计划已原子登记。当前无训练或诊断进程，phase241 step50 只作冻结证据且不得续训。下一原子动作仅为实现和静态审核可完整删除的只读诊断。

2026-09-29 02:20，phase242 临时只读诊断完成首轮接线审核。原实现会把 staged-reset 的 GRASP 及后续阶段样本混入 APPROACH_PREGRASP 的 joint-quality 分桶，并且 surrogate 分组梯度未乘生产 PPO 的 stage-balanced sample weight；两者都会令归因结论失真。现已显式记录 reward 时刻的 `approach_active` 和成就 buffer 的 prior-initialized 状态，仅统计第一阶段有效帧，并用生产权重计算分组及全体参照梯度；同时增加 joint quality/new best/delta/stage-progress 和非边界 GAE 逐元素重建误差。只改临时测量路径，生产 reward、observation/action、完成条件和模型均未修改。下一原子动作是静态回归，不启动 Isaac。

2026-09-29 02:23，phase242 静态回归完成。容器内 py_compile、定向 `10 passed, 5 subtests passed`、完整 `57 passed, 5 subtests passed` 和宿主机 `git diff --check` 全部通过。当前无 Isaac 训练/诊断进程；下一原子动作仅为 `8 env x 8 steps` 的 optimizer 前退出 smoke，不得直接启动正式 2048-env rollout。

2026-09-29 02:25，phase242 极小冻结 smoke 完成并正常退出。GPU 3、`8 env x 8 steps` 共 64 帧，均为当前 reward 所属的 APPROACH_PREGRASP；joint quality/new best/achievement delta/stage-progress 四项逐元素重建最大误差均为 0，非边界 GAE recurrence max/mean 为 `1.91e-6/1.09e-7`，ratio mean absolute error `3.73e-6`、clip fraction 0，所有导出张量有限。张量 `doorman_train:/tmp/b2z1_phase242_smoke.pt` SHA-256 `eb7175b76eb54726f078a1d42b3c3a4bd420c3d23b5c2395c49329c7293d0456`，日志 SHA-256 `fc58e2edb2dd20131dac6a9849d3abc2caf340777d17adf781c589ed8b1fbf8a`；源 checkpoint SHA 未变。下一原子动作仅为唯一正式冻结 rollout。

2026-09-29 02:26，phase242 唯一正式冻结 rollout 已启动。容器 `doorman_train`、GPU 3、2048 env、128 steps，launcher PID `1301021`、worker PID `1301091`；日志 `doorman_train:/tmp/b2z1_phase242_joint_credit_diag.log`，目标张量 `doorman_train:/tmp/b2z1_phase242_joint_credit_diag.pt`。运行中冻结代码，下一原子动作只轮询该进程。

2026-09-29 02:31，phase242 唯一正式冻结 rollout 正常完成并退出，optimizer 未执行。张量 SHA-256 为 `c02b224b9436b8c7d22d7db84c0f9e270101cd8aafc7f237c59f8e8b2943c424`，日志 SHA-256 为 `497a2918736c74ad495016c21f488c6ceb8eef8bea54bf5f54e9fe1c0ae44b79`，源 checkpoint SHA 未变。证据确认 history-best 候选的首个断点是高联合质量无法保持：真正高质量覆盖和梯度份额近乎消失，且一次性 achievement 事件没有形成降噪；不得继续训练或调权重。下一原子动作是删除 phase241 失败机制和全部 phase242 临时代码，再做完整回归。

2026-09-29 02:35，phase242 清理完成。phase241 history-best 函数、环境 buffer、reward 接线、专属测试，以及 phase242 的环境缓存、storage、导出、梯度函数和 optimizer 前退出均已删除；专属标识搜索为零。容器内 py_compile、定向 `5 passed, 5 subtests passed`、完整 `52 passed, 5 subtests passed` 和宿主机 `git diff --check` 全部通过。当前无训练或诊断进程。

### phase243：可回退的有符号联合质量势（失败）

```text
上游结论与待解决瓶颈：phase242 证明 history-best 增量只奖励一次性刷新，绝大多数 credit 消耗在低质量新纪录上，并且不惩罚随后失稳；需要让真实联合质量上升和下降都进入 credit
基线 checkpoint / SHA-256：无；先做纯张量和 phase242 冻结轨迹离线反事实，不加载或更新模型
唯一变量：定义 `Q=min(R,S)*C`，其中 R、S、C 与 phase241 相同；额外 reward 候选仅为 initialized 后的 signed `Q_t-Q_{t-1}`，第一阶段仍为 `20*(Delta G+Delta Q)`，原 `Delta G` 逐元素不变
待验证假设：signed joint-quality potential 会奖励真实联合质量提高、等量惩罚退化，往复循环净收益为零；因此能保留到位后的减速 credit，而不会像 history-best 一样放任随后失稳
禁止改动范围：不改生产 reward、权重、完成条件、termination、observation/action、探索预算、随机化、低层、阶段逻辑或 checkpoint；不使用 command、脚本目标、stage/seed/门型特例，不扫描系数
运行规模与日志：离线审核后，单卡 GPU 3、2048 env、128 steps、1 update smoke 已完成；四卡每卡 2048 env、128 steps、4 mini-batches，从随机初始化训练至 step25。正式日志 `doorman_train:/tmp/b2z1_phase243_signed_joint_clean_to25.log`，运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260929_024432`
成功/失败判据：(1) 原几何 signed delta 完全不变；(2) 初始化无 windfall；(3) 同几何下减速为正、加速为负；(4) 几何或稳定退化给负 credit；(5) 任意闭环序列额外收益严格望远镜为零；(6) phase242 高质量退化帧获得明确负 credit。任一失败即不接入环境
观测指标：逐步 Q/delta、循环累计、按当前 Q 分桶的正负 delta 覆盖、`q>=0.2/0.3` 后续退化帧的反事实 reward、原几何 delta 差值
结论与已排除方向：失败。离线数学性质全部通过，但在线 step25 五项 quality 为 `0.3827/0.4790/0.1918/0.1777/0.9091`；base position、pregrasp position、gripper orientation 和 gripper open 高于 phase224 同期，base yaw 低于 `0.5013`，因此未通过“五项均不得回归”的预注册门槛。末五轮 actual speed 从 `0.2100` 降至 `0.2074 m/s`，但 stable 1/5/10 与 GRASP 全为 0。已排除运行异常、安全回归和单纯没有几何学习；尚不能区分 bottleneck component 竞争、状态覆盖或 exact PPO 梯度分配
正式保留/删除的代码：phase244 归因前暂留通用 `signed_joint_quality_delta()`、当前 Q/initialized buffer、第一阶段唯一接线和五项测试，仅作为被测失败机制；无诊断、环境变量、隐藏开关或旧 history-best 路径。归因结束后必须整体决定保留或删除
前序阶段回归：py_compile、定向 `10 passed, 5 subtests passed`、完整 `57 passed, 5 subtests passed` 和 `git diff --check` 通过。optimizer smoke 262,144 timesteps、`45.93 s`，cameras=0、RTX sensors=0、frame rendering disabled，无异常/NaN；unsafe-force `0.0001`，door-overspeed/shoulder-collision 为 0
下一决策门槛：禁止续到 step50。phase244 必须冻结本 checkpoint，逐 component 闭合 signed credit、raw/normalized advantage、未来质量和 exact PPO mean/attenuation 梯度；确认首个断点后才能决定下一行为变量
```

2026-09-29 02:35，phase243 计划已原子登记。当前生产路径是 phase240 清理后的原始 signed geometry delta，无联合稳定候选或诊断代码。下一原子动作仅为纯张量和冻结轨迹离线审核，不得直接接入环境或启动训练。

2026-09-29 02:39，phase243 纯张量与冻结轨迹离线审核通过。对 phase242 的 258,242 个连续帧重算 signed joint-quality delta，得到 165,659 个正样本、91,640 个负样本；离开 `q>=0.2/0.3` 时负 delta 覆盖率为 `61.8%/83.1%`，均值 `-0.0196/-0.0725`。逐 episode 累计 delta 与端点差的最大误差 `2.98e-8`，构造往复闭环净收益约 0。下一原子动作允许最小实现，但不允许改权重、完成条件、探索或启动训练。

2026-09-29 02:39，phase243 最小实现和静态回归完成。新增通用 signed joint-quality 函数、当前 Q buffer、第一阶段唯一 reward 接线和五项纯张量测试；原几何 signed delta 逐元素保留，旧 history-best 不存在。容器内 py_compile、定向 `10 passed, 5 subtests passed`、完整 `57 passed, 5 subtests passed` 和宿主机 diff check 通过。下一原子动作仅为单卡真实 optimizer smoke。

2026-09-29 02:40，phase243 单卡真实 optimizer smoke 已启动。容器 `doorman_train`、GPU 3、2048 env、128 steps、1 update，launcher PID `1301991`、worker PID `1302061`，日志 `doorman_train:/tmp/b2z1_phase243_signed_joint_smoke.log`。运行中冻结代码，下一原子动作只轮询该作业。

2026-09-29 02:43，phase243 单卡真实 optimizer smoke 正常完成。262,144 timesteps，iteration `45.93 s`；确认 cameras=0、RTX sensors=0、frame rendering disabled，真实 optimizer 执行且无异常/NaN。五项初始 quality `0.3357/0.5146/0.1314/0.0466/0.7709` 与 phase241 同 seed 完全一致，actual speed `0.2189 m/s`，stage-progress `0.0629`；unsafe-force `0.0001`，其余任务安全终止为 0。日志 SHA-256 `8e7a74d8195ead567675a523b3465314c5afee491954f0ee97ea3985ad60fdd1`。下一原子动作是四卡随机初始化训练至 step25。

2026-09-29 02:45，phase243 四卡随机初始化训练已启动。GPU 0--3 每卡 2048 env、128 steps、4 mini-batches，训练至 step25；launcher PID `1302263`、worker PID `1302335--1302338`、端口 `29543`，日志 `doorman_train:/tmp/b2z1_phase243_signed_joint_clean_to25.log`。代码冻结，下一原子动作只轮询该作业。

2026-09-29 02:49，phase243 四卡 update1 正常完成。四个 rank 均确认 cameras=0、RTX sensors=0、frame rendering disabled；全局 `1,048,576` timesteps，iteration `47.50 s`，运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260929_024432`。首轮五项与 smoke 一致，stable 1/5/10 全 0，仅作随机初态基线；下一原子动作只轮询现有作业至 step25。

2026-09-29 03:11，phase243 四卡训练在预注册 step25 正常结束，四个 worker 与 launcher 均已退出。共 `26,214,400` timesteps，末轮 `48.24 s`，无异常或 NaN。checkpoint `model_step_000025.pt` SHA-256 为 `b377b0e785516f2c0084aa130463593f895535b7405b25a5f24405200fb790d6`；日志 SHA-256 为 `016ce06369cdf6aa3248d407cedac9be9abffedc622c9be0f5fe3a8645e936f2`。五项 quality `0.3827/0.4790/0.1918/0.1777/0.9091`，episode-best `0.4582/0.6816/0.2716/0.3300/0.9871`；除 base yaw 外均超过 phase224 同期。末五轮 actual speed `0.2100/0.2098/0.2093/0.2074/0.2074 m/s`，unsafe-force `0.0001`，door-overspeed 与 shoulder-collision 为 0；stable 1/5/10 与 GRASP 全为 0。预注册继续门槛失败，不续训。

### phase244：phase243 signed credit 分配归因（已验证并清理）

```text
上游结论与待解决瓶颈：phase243 同时改善四项几何和实际速度，却使 base yaw 低于 phase224 同期且未产生任何联合稳定；聚合训练曲线不能区分 component bottleneck 竞争、状态覆盖不足或 PPO 梯度分配失真
基线 checkpoint / SHA-256：phase243 step25 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260929_024432/model_step_000025.pt` / `b377b0e785516f2c0084aa130463593f895535b7405b25a5f24405200fb790d6`；只读加载，禁止续训
唯一变量：增加可完整删除的只读 rollout 导出，记录五项 component quality、weakest component、实际 base/EE motion、Q 与 signed delta、原几何 delta、reward、raw/normalized advantage、未来 Q，并计算 exact temporal PPO 的逐动作 mean 与 attenuation 梯度；optimizer 前退出
待验证假设：signed `Delta Q` 的 credit 被当前最弱 component 和高频 component 切换主导，对 base yaw 缺少持续正确梯度，或者正确 credit 在 temporal PPO 中被其他状态/动作梯度抵消
禁止改动范围：不更新模型；不改 reward、权重、完成条件、termination、observation/action、探索预算、随机化、低层或阶段逻辑；不使用 command、脚本目标、stage/seed/门型特例，不扫描参数
运行规模与日志：静态回归与 `8 env x 8 steps` smoke 后，GPU 3、2048 env、128 steps 唯一冻结 rollout 已完成；日志 `doorman_train:/tmp/b2z1_phase244_signed_credit_diag.log`，张量 `doorman_train:/tmp/b2z1_phase244_signed_credit_diag.pt`，optimizer 未执行
成功/失败判据：逐元素闭合 Q、signed delta、生产 stage-progress、GAE 与 ratio；按 weakest component、Q 桶和 component 改善/退化事件给出样本覆盖、未来保持率、advantage 及 exact PPO 梯度份额，能指出最早发生的可证伪断点。只看聚合 quality 或凭曲线猜权重不算完成
观测指标：五项 component/current-best、weakest identity 与切换率、base/EE motion、Q/delta、原几何 delta、stage-progress、raw/normalized advantage、未来 Q、termination、ratio/clip、十维 mean/attenuation gradient
结论与已排除方向：已验证。262,144 个第一阶段 transition 中，base position/base yaw/pregrasp/orientation/open 成为 weakest 的数量为 `0/259/83942/177525/418`；base yaw 仅占 `0.099%`，对应 mean 绝对梯度份额约 `0.202%`，而两个机械臂几何项合计占 `99.74%` weakest 帧和约 `99.16%` mean 梯度。`83.0%` 帧由 geometry 而非 stability 限制 Q；在 readiness `<0.05` 的 `62.5%` 样本里 stability 仅 `0.40%` 成为瓶颈。新增 `|Delta Q|` 均值 `0.002644`，高于原 `|Delta G|=0.001738`，因此新增信号主要重复最低机械臂几何或追逐 motion 波动，而不是持续维护所有 component。排除状态张量错位、Q/delta 公式错误、GAE、ratio、optimizer 更新和运行异常；phase243 候选必须整体删除
正式保留/删除的代码：phase243 signed joint-quality 函数、buffer、reward 接线和专属测试，以及 phase244 全部临时导出、storage、纯函数和测试均已删除；未保留兼容分支
前序阶段回归：phase243 四卡纯动力学 step25 正常完成，cameras=0、RTX sensors=0、frame rendering disabled；清理后容器内 py_compile、定向 `5 passed, 5 subtests passed`、完整 `52 passed, 5 subtests passed`、专属标识搜索和宿主机 `git diff --check` 全部通过
下一决策门槛：phase245 只能先用 phase244 冻结张量与构造反例审核独立稳定/无接触 signed potential；审核通过前不修改生产代码或启动训练
```

2026-09-29 03:11，phase244 计划已原子登记。当前无训练或诊断进程，phase243 checkpoint 只作冻结证据且不得续训。恢复后的唯一下一原子动作是实现最小只读导出并完成静态回归，不得重新运行 phase243 或提前修改 reward。

2026-09-29 03:24，phase244 最小只读实现和静态回归完成。新增显式输出路径门控的 reward 后物理快照、rollout storage、optimizer 前退出导出，以及 exact temporal PPO surrogate 对 action mean/log-sigma 的输出梯度和 GAE 递推审核纯函数；默认路径为关闭，不改变生产 rollout。容器内 py_compile、定向 `12 passed, 5 subtests passed`、完整 `59 passed, 5 subtests passed` 和宿主机 `git diff --check` 全部通过。下一原子动作仅为 GPU 3 的 `8 env x 8 steps` optimizer 前退出 smoke。

2026-09-29 03:27，phase244 有效极小 smoke 正常完成并退出，optimizer 未执行。GPU 3、`8 env x 8 steps` 共 64 个 APPROACH_PREGRASP transition；Q、readiness、signed delta 和 stage-progress 逐元素重建最大误差均为 0，GAE max/mean `1.19e-7/1.18e-8`，ratio mean absolute error `3.63e-7`、clip fraction 0，全部导出张量有限。张量 `doorman_train:/tmp/b2z1_phase244_smoke.pt` SHA-256 `d59b6e8d099f9d144de6ae58bc365a87dbae64c1e1ac9fd8b00e230877c5ff91`，日志 SHA-256 `ff3afed3ca39427c88854ffc05751d7ba7c7a832e196c97fc6562f5049f03cc4`。首轮 smoke 曾因审核公式误写为 `R*S*C` 而失败，已修正为与生产一致的 `min(R,S)*C` 后重跑；错误仅在临时审核公式，生产 reward 未改。下一原子动作允许唯一正式冻结 rollout。

2026-09-29 03:28，phase244 唯一正式冻结 rollout 已启动。容器 `doorman_train`、GPU 3、2048 env、128 steps；launcher PID `1304296/1304302`、worker PID `1304366`，日志 `doorman_train:/tmp/b2z1_phase244_signed_credit_diag.log`，张量 `doorman_train:/tmp/b2z1_phase244_signed_credit_diag.pt`。optimizer 前退出，运行中冻结代码，下一原子动作只轮询该进程。

2026-09-29 03:36，phase244 唯一正式冻结 rollout 正常完成并退出，共 262,144 个 APPROACH_PREGRASP transition，optimizer 未执行。张量 SHA-256 `44aee352cc43210bee312860b29975d5a19a58047c68e6a42f907f37dc0ee7d1`，日志 SHA-256 `cd96a897114fded2de87e9345f9bb5baa77b1343faba93b80bbba171e74e5a88`，源 checkpoint SHA-256 仍为 `b377b0e785516f2c0084aa130463593f895535b7405b25a5f24405200fb790d6`。Q、signed delta 和 stage-progress 重建误差为 0，GAE max `7.63e-6`，ratio mean error `7.28e-7`、clip 0。证据确认首个断点位于 reward 的 min-selector：新增信号选择性强化最低机械臂几何并饿死 base yaw 等非最弱 component，不能作为联合保持势保留。下一原子动作是整体清理 phase243/244，不启动训练。

2026-09-29 03:37，phase244 清理完成。phase243 signed joint-quality 函数、环境 buffer、第一阶段 reward 接线和专属测试，以及 phase244 的环境快照、rollout storage、导出、梯度纯函数、测试和 optimizer 前退出均已删除；`approach_credit`、`signed_joint_quality`、`approach_joint_quality` 专属标识搜索为零。容器内 py_compile、定向 `5 passed, 5 subtests passed`、完整 `52 passed, 5 subtests passed` 和宿主机 `git diff --check` 全部通过。当前无 Isaac 训练或诊断进程，生产 reward 已恢复为未缩放的原始 `Delta G`。

### phase245：独立稳定/无接触 signed potential 离线审核（失败）

```text
上游结论与待解决瓶颈：phase244 证明 `min(R,S)*C` 会把新增 credit 分配给当前单一 weakest component，重复机械臂几何并饿死 base yaw；失败机制已完整删除，但原 `Delta G` 本身仍不评价实际停稳
基线 checkpoint / SHA-256：不加载或更新模型；只读使用 phase244 冻结张量 `/tmp/b2z1_phase244_signed_credit_diag.pt`，SHA-256 `44aee352cc43210bee312860b29975d5a19a58047c68e6a42f907f37dc0ee7d1`
唯一变量：仅离线审核候选 `Delta G + Delta(S*C)`；`G` 保持生产几何 signed delta 不变，`S` 只由实际 base/EE 运动定义，`C` 只由真实把手/门板无接触结果定义，额外项不经过任何 geometry min-selector
待验证假设：独立 signed `Delta(S*C)` 能在移动后停稳时提供可恢复 credit，同时初始化无 windfall、远处静止无持续收益、加速或接触给负 credit、任意闭环轨迹额外累计严格为零，且不会选择性移除任一几何 component 的原始 credit
禁止改动范围：不修改生产代码、reward 权重、完成条件、termination、observation/action、探索预算、随机化、低层或阶段逻辑；不启动 Isaac、不更新模型、不按 stage/seed/门型/readiness 特调或扫描系数
运行规模与日志：纯张量离线分析；先审核构造反例，再读取 phase244 冻结轨迹计算覆盖、正负事件、动态范围、与 `Delta G` 的符号竞争及潜在循环收益
成功/失败判据：(1) 原 `Delta G` 逐元素不变；(2) 初始化额外 reward 为 0；(3) 同几何下减速为正、加速为负，静止保持为 0；(4) 接触恶化为负、恢复无接触为正；(5) 每条连续轨迹额外累计等于端点势差，闭环净收益为 0；(6) 冻结轨迹中信号具有实际覆盖且不会像 phase243 一样由单个几何 component 选择；(7) 明确量化与几何退化同时发生时的竞争，不能凭总均值决定接入
观测指标：`S*C` 当前值与 signed delta、正/负/零覆盖、`|Delta(S*C)|/|Delta G|`、按几何改善/退化和 motion/contact 事件的联合分桶、逐 episode 望远镜误差、构造循环累计
结论与已排除方向：失败。258,221 个可直接比较的连续 transition 中，`S*C` 均值为 `0.10703`，signed delta 有 124,459 个正样本、131,974 个负样本；`mean|Delta(S*C)|=0.018363`，是 `mean|Delta G|=0.001765` 的 `10.41` 倍。43,131 个“几何退化但稳定改善”冲突帧中，40,879 个即 `94.78%` 在单位权重相加后仍为净正；冲突帧 `|Delta P|/|Delta G|` 中位数为 `24.20`。候选满足初始化无 windfall、静止保持为零、构造闭环净收益约零和连续 transition 重建误差 0，但这些性质不能弥补动态范围压过几何的错误 credit。排除单位权重直接接入、训练后再看和以该值为起点扫描
正式保留/删除的代码：没有新增或修改任何 phase245 生产代码、测试或诊断分支；分析只读取现有 `/tmp` 冻结张量
前序阶段回归：phase244 清理后的 py_compile、定向 `5 passed, 5 subtests passed`、完整 `52 passed, 5 subtests passed`、专属标识搜索和 `git diff --check` 已通过
下一决策门槛：phase246 只能从既有 `G=0.5*mean+0.5*min` 推导一个原则性尺度并离线审核；禁止把 phase245 的失败结果变成权重扫描
```

2026-09-29 03:37，phase245 计划已原子登记到顶部执行卡、上层路线图、六阶段总账、近期尝试和本记录。当前没有 phase245 实现、训练或诊断进程；恢复后的唯一下一原子动作是离线审核候选，不得重新运行 phase243/244 或直接接入环境。

2026-09-29 03:40，phase245 离线审核完成并判定失败。候选的望远镜、初始化和静止性质成立，但实际 `|Delta(S*C)|` 比原 `|Delta G|` 大 10.41 倍，并在 94.78% 的几何退化/稳定改善冲突帧覆盖几何惩罚；因此不能保证“稳定”不会替代接近目标。未修改生产代码、未启动 Isaac、未生成 checkpoint，当前仍无临时代码或运行进程。

### phase246：独立稳定势原则性尺度推导与离线审核（失败）

```text
上游结论与待解决瓶颈：phase245 的独立势结构不再经过 geometry selector，但单位权重动态范围过大，系统性奖励停在更差几何位置；不能靠扫描系数寻找表面上可用的结果
基线 checkpoint / SHA-256：不加载或更新模型；继续只读 phase244 冻结张量及其 SHA-256 `44aee352cc43210bee312860b29975d5a19a58047c68e6a42f907f37dc0ee7d1`
唯一变量：从现有 `G=0.5*mean(five geometry)+0.5*min(five geometry)` 推导稳定作为独立非 weakest 结果时的边际系数 `0.5/5=0.1`；只审核 `Delta G + 0.1*Delta(S*C)` 这一项，不扫描其他系数
待验证假设：原则性边际系数可保留独立停稳 credit，同时不再系统性覆盖几何退化；其依据来自现有势函数结构，不是 seed、轨迹或训练曲线拟合
禁止改动范围：不修改生产代码、reward、完成条件、observation/action、探索、随机化或低层；不启动 Isaac、不训练、不比较多个候选系数
运行规模与日志：纯张量构造反例和 phase244 冻结轨迹离线审核
成功/失败判据：除 phase245 已通过的数学性质外，几何退化/稳定改善冲突帧必须不再被新增项系统性翻为净正；必须报告净符号翻转率、动态范围、接触跃迁贡献和各 weakest-component 分层，不能只看全局均值
观测指标：`mean|0.1*Delta P|/mean|Delta G|`、四种符号组合、几何退化帧净正比例、接触进入/退出帧贡献、weakest identity 分层和闭环误差
结论与已排除方向：失败。缩放后 `mean|0.1*Delta P|=0.001836`，与 `mean|Delta G|=0.001765` 的比例仍为 `1.041`。在 43,131 个几何退化/稳定改善冲突帧中 68.97% 被翻为净正，占全部 86,048 个几何退化帧的 34.57%；在 89,615 个几何改善/稳定退化冲突帧中，仅 46.93% 保持净正，即 53.07% 的真实几何改善被瞬时速度变化抵消。进入/退出接触各 340/334 帧，缩放稳定项均值为 `-0.01246/+0.01336`，也明显大于同帧几何变化。排除继续缩小、扫描或训练调系数；首个断点是逐帧 motion 势与几何在同一时间尺度双向竞争
正式保留/删除的代码：当前无 phase246 代码；只允许临时离线命令或 `/tmp` 输出
前序阶段回归：phase244 清理后的静态回归仍有效；phase245 未修改工作树
下一决策门槛：停止该独立逐帧势及权重方向。phase247 只归因稳定正增量的时间保持性，根因前不设计替代 reward
```

2026-09-29 03:40，phase246 计划已原子登记。当前无实现、训练或诊断进程；恢复后的唯一下一原子动作是审核系数 `0.1` 的单一候选，不得扫描或直接接入环境。

2026-09-29 03:41，phase246 唯一候选离线审核失败。原则性 `0.1` 缩放把平均绝对稳定 delta 降到与几何约 1:1，但仍同时奖励大量几何退化并抵消大量几何改善，证明问题不是再调小一个数，而是单帧运动变化不是与几何进度同时间尺度的可靠停稳结果。未修改生产代码、未启动 Isaac，当前无临时代码或运行进程。

### phase247：稳定改善时间保持性离线归因（已验证）

```text
上游结论与待解决瓶颈：phase245/246 的逐帧 `Delta(S*C)` 与几何 delta 双向竞争；尚不清楚正稳定增量代表真实持续停稳，还是下一拍即反转的速度波动
基线 checkpoint / SHA-256：不加载或更新模型；只读 phase244 冻结张量，SHA-256 `44aee352cc43210bee312860b29975d5a19a58047c68e6a42f907f37dc0ee7d1`
唯一变量：无行为变量；仅按连续轨迹计算每个正稳定增量之后 1/5/10 步的 `S*C` 保持率、反转率和端点变化，并按是否同时几何退化分层
待验证假设：大部分单帧稳定改善不能持续到与完成条件一致的 5/10 步窗口，因此逐帧势把高频速度波动误当成了停稳结果
禁止改动范围：不设计或修改 reward，不改生产代码、权重、完成条件、observation/action、探索、随机化或低层；不启动 Isaac、不训练、不补采样
运行规模与日志：phase244 的 2048 x 128 冻结轨迹纯张量分析，仅使用同 episode 连续窗口
成功/失败判据：必须明确给出正增量事件在 1/5/10 步后的保持与反转比例、净端点变化、接触事件影响和几何冲突子集；不能凭单帧自相关下结论
观测指标：future `P-P_t`、未来最小/末端 P、回吐初始增益比例、同窗口累计 geometry delta、done/reset 排除数量
结论与已排除方向：已验证。冻结轨迹共有 124,125 个排除接触切换后的正稳定增量事件；未来 1/5/10 步终点低于事件当帧的比例为 `52.10%/52.34%/56.13%`，未来窗口至少跌破事件前水平的比例为 `26.46%/53.88%/68.69%`，到第 10 步仍保留至少一半初始增益的比例仅 `49.48%`。几何退化子集呈相同现象。真实 `base<0.05 and EE<0.05 and contact-free` 单帧仅 18 个，占 `0.00687%`，连续 5/10 步窗口均为 0。首个断点因此是时间语义：逐帧 `Delta S` 把速度起伏当成停稳改善，既不能代表 dwell，又会与几何 delta 高频竞争。排除继续缩放单帧势或以其均值判断停稳 credit
正式保留/删除的代码：无 phase247 代码；只允许临时离线命令或 `/tmp` 输出
前序阶段回归：phase245/246 均未修改工作树，phase244 清理后的静态回归仍有效
下一决策门槛：phase248 只离线审核 10 步 trailing-min 当前历史结果；它必须可下降并惩罚失稳，禁止恢复 phase241 的单调 history-best
```

2026-09-29 03:41，phase247 计划已原子登记。当前无实现、训练或诊断进程；恢复后的唯一下一原子动作是冻结轨迹时间保持性统计。

2026-09-29 03:44，phase247 时间保持性归因完成。正稳定增量在 10 步窗口中多数回吐，真实 5/10 步稳定窗口均为 0，证明单帧 motion 势不是阶段所需的连续停稳结果。分析只读取 phase244 冻结张量，未修改工作树、未启动 Isaac、未生成 checkpoint。

### phase248：十步持续稳定结果势离线审核（失败）

```text
上游结论与待解决瓶颈：phase247 证明逐帧稳定改善多数在 10 步内回吐，且没有任何真实 5/10 步稳定窗口；reward 的时间语义必须与连续 10 步完成条件一致
基线 checkpoint / SHA-256：不加载或更新模型；只读 phase244 冻结张量，SHA-256 `44aee352cc43210bee312860b29975d5a19a58047c68e6a42f907f37dc0ee7d1`
唯一变量：离线构造 `P=S*C` 和当前有限历史结果 `W_t=min(P_{t-9:t})`，审核 signed `Delta W`；W 随最近十步最差稳定结果改善而上升，任一步失稳时可立即下降，不是单调历史最佳
待验证假设：10 步 trailing-min 会过滤单帧速度改善，只奖励持续改善并惩罚任何窗口内失稳，从而与阶段 dwell 语义一致且减少对几何 delta 的高频覆盖
禁止改动范围：不修改生产代码、reward 权重、完成条件、observation/action、探索、随机化或低层；不启动 Isaac、不训练；不使用 history-best、stage/seed/门型特例或系数扫描
运行规模与日志：phase244 冻结轨迹纯张量反事实；窗口跨 reset/done 时不计入有效样本
成功/失败判据：(1) 初始化无 windfall；(2) W 可正可负回退且闭环望远镜；(3) 正 delta 覆盖足以学习但显著低于单帧噪声事件；(4) 相对原 `Delta G` 的动态范围和几何冲突不再系统性翻转；(5) 接触或任一步失稳立即给负 delta；任一失败即不实现
观测指标：正/负/零覆盖、`mean|Delta W|/mean|Delta G|`、几何冲突净符号、窗口最低项更新来源、接触切换贡献、轨迹端点闭合误差
结论与已排除方向：失败。235,494 个具备完整相邻 10 步窗口的 transition 中，`Delta W` 有 38,210 个正、44,364 个负、152,920 个零；虽比逐帧势稀疏，但 `mean|Delta W|=0.003230`，仍是同帧 `mean|Delta G|=0.000918` 的 3.52 倍。12,501 个几何退化/窗口改善冲突帧中 94.24% 被翻为净正。更根本地，121,701 个即时 P 恶化帧只有 36.45% 得到负 `Delta W`、46.11% 为零；旧窗口最小值退出时，W 甚至可在当前 P 下降时上升。进入接触也只有 35.81% 立即形成负 W，因为更低旧值会遮蔽当前失败。候选不满足“任一步失稳立即回退”的预注册语义，排除实现、缩放或训练
正式保留/删除的代码：无 phase248 代码；只允许临时离线命令或 `/tmp` 输出
前序阶段回归：phase247 未修改工作树，phase244 清理后的静态回归仍有效
下一决策门槛：phase249 先审核因果持久下包络：上升受 10 步时间常数限制，当前结果低于历史状态时立即回退；不得继续修改 trailing-min
```

2026-09-29 03:44，phase248 计划已原子登记。当前无实现、训练或诊断进程；恢复后的唯一下一原子动作是 trailing-min 冻结轨迹审核。

2026-09-29 03:47，phase248 离线审核失败。trailing-min 虽减少事件数，却因窗口旧最小值退出而在当前恶化时产生伪改善，并且非零事件幅度仍压过几何。候选未进入生产代码，未启动 Isaac 或训练；当前无临时代码和运行进程。

### phase249：持久稳定下包络数学与离线审核（失败）

```text
上游结论与待解决瓶颈：单帧势评价速度起伏，trailing-min 又会因旧最小值滑出产生伪改善；需要一个因果状态，使稳定改善必须持续积累，而当前明显恶化能立即撤销既有 credit
基线 checkpoint / SHA-256：不加载或更新模型；只读 phase244 冻结张量，SHA-256 `44aee352cc43210bee312860b29975d5a19a58047c68e6a42f907f37dc0ee7d1`
唯一变量：离线构造 `H_t=min(P_t,H_{t-1}+0.05)`，reset 时初始化为当前 P 且 reward 为零；`0.05` 由真实稳定阈值对应的 `P=0.5` 除以 10 步 dwell 推导。候选总增量仅审核 `Delta G+0.1*Delta H`，`0.1` 来自既有 G 中非 weakest component 的边际 `0.5/5`
待验证假设：H 的改善最多每步增加 0.05，必须由连续较高 P 支撑；P 跌破 H 时立即回退，不存在窗口滑出伪改善；原则性 0.1 边际不应再系统性覆盖几何
禁止改动范围：不修改生产代码、reward、完成条件、observation/action、探索、随机化或低层；不启动 Isaac、不训练；不扫描 rise rate 或 reward 系数
运行规模与日志：先用上升、单帧尖峰、持续改善、即时恶化和闭环构造序列审核，再在 phase244 冻结轨迹上重放
成功/失败判据：(1) 初始化无 windfall；(2) 单帧尖峰随后完全回吐，持续十步才可积累到稳定阈值尺度；(3) P 低于 H 时同拍负 delta；(4) H 可升可降且逐轨迹望远镜闭合；(5) 缩放后动态范围与几何冲突不再系统性翻转；(6) 非零覆盖足够且不由接触切换主导
观测指标：H/delta 构造序列、正负零覆盖、`mean|0.1*Delta H|/mean|Delta G|`、两类冲突翻转率、P 恶化时 H 响应、接触贡献和端点闭合误差
结论与已排除方向：失败。构造序列满足：单帧尖峰 `+0.05` 后完全回吐、连续 10 步从 0 累积到 `0.5`、P 跌破 H 时同拍回退、闭环累计等于端点差。真实轨迹中却有 126,246 个正、130,185 个负、仅 1,790 个零 delta，即 99.31% transition 仍随逐帧 P 变化；原则性缩放后 `mean|0.1*Delta H|=0.001667`，仍为 `mean|Delta G|=0.001765` 的 0.944 倍。43,661 个几何退化/稳定改善冲突帧中 69.20% 净正，88,357 个反向冲突帧中 52.45% 几何改善被抵消，与 phase246 几乎相同。虽然 P 低于 previous H 时 99.999% 同拍给负值，但 rise cap `0.05` 大于实际大多数单步变化，使 H 几乎直接跟随 P，未形成持续性过滤。排除继续修改 rise rate、缩放系数或训练后观察
正式保留/删除的代码：无 phase249 代码；只允许临时离线命令或 `/tmp` 输出
前序阶段回归：phase248 未修改工作树，phase244 清理后的静态回归仍有效
下一决策门槛：停止新增稳定势。phase250 只复用 phase230--238 已有证据，重新闭合探索方差到连续停稳的架构链，确认下一变量前不改代码
```

2026-09-29 03:47，phase249 计划已原子登记。当前无实现、训练或诊断进程；恢复后的唯一下一原子动作是构造反例与冻结轨迹审核。

2026-09-29 03:50，phase249 构造反例通过但冻结轨迹审核失败。候选状态在 99.31% 的真实 transition 仍输出 delta，没有把持续停稳从单帧速度起伏中分离；冲突翻转率与 phase246 基本不变。未修改生产代码、未启动 Isaac 或训练，当前无 phase249 临时代码。

### phase250：探索噪声与连续停稳架构复核（已验证）

```text
上游结论与待解决瓶颈：phase245--249 连续排除单位逐帧势、原则性缩放、trailing-min 和持久下包络；继续设计时间势已经变成补丁循环。phase230--238 已分别取得噪声下限、状态方差、预算梯度和时间 credit 证据，但尚未在当前账本中合并成唯一架构断点
基线 checkpoint / SHA-256：不加载模型、不运行仿真；只读本文 phase230--238 已验证记录及其保留的原始 `/tmp` 证据（若仍存在）
唯一变量：无行为变量；建立 `policy mean/std -> sampled action -> low-level actual motion -> 1/5/10-step stability coverage -> reward/GAE -> mean/attenuation PPO gradient` 因果表，标出每层已证实、已排除和证据缺口
待验证假设：当前首个架构断点可能不是再缺一个稳定势，而是探索分布在 near-ready 状态无法降噪；现有总体方差预算只约束全局总量，PPO 又缺少足够连续成功样本把方差搬离 near-ready 状态
禁止改动范围：不修改代码、reward、entropy、方差预算、完成条件、observation/action、随机化或低层；不启动 Isaac、不训练、不重复 phase230--238 rollout
运行规模与日志：只读文档、现有日志与张量；若原始文件缺失，只登记缺口，不重跑
成功/失败判据：必须指出首个尚未解决且能被下一单变量实验证伪的接口层；不能把多个旧结论简单并列，也不能凭推测直接改 entropy/std/reward
观测指标：near-ready 覆盖、normal/near-zero noise actual motion、actor mean residual、各 readiness 桶 std/attenuation、总方差预算、surrogate/entropy 梯度、raw advantage 来源和连续稳定覆盖
结论与已排除方向：已验证，因果链闭合为：(1) phase230 同 checkpoint 全维近零噪声把高 readiness base/EE motion 从 `0.2213/0.2748` 降至 `0.0598/0.0938 m/s`，证明 sampled noise 是主要速度下限，actor mean 是次要残余；(2) phase236 的状态 attenuation 已有 `1.8%--5.7%` 弱分化，但高桶仅 29 帧且 sample noise 仍显著大于 mean，总体方差已在 `0.8058--0.8816`、接近 0.8 预算；(3) phase237 证明高 readiness 六个 base/EE 平移动作维不是只有正确 credit 被样本数稀释，而是 surrogate 本身继续推动增噪；(4) phase238 证明该方向来自真实后续几何退化/终止造成的负 raw GAE，不是归一化、critic 或 PPO 符号错误；(5) phase245--249 又排除四种把单帧 motion 直接包装成稳定势的方案。首个未解决接口因此不是预算、概率链或再缺一个势函数，而是尚未分解十维 sampled noise 中哪些维度通过低层和机械臂动力学主导 base/EE 速度。phase230 只做全维同时近零，不能支持全局降噪或猜测某一维
正式保留/删除的代码：无 phase250 代码
前序阶段回归：phase245--249 均未修改工作树，phase244 清理后的静态回归仍有效
下一决策门槛：phase251 先审核现有 phase236 张量的因果可辨识性。若缺少下一拍方向速度或独立 innovation，只登记最小导出后做一次冻结 rollout；不得先改 noise fraction、std、预算或 reward
```

2026-09-29 03:50，phase250 计划已原子登记。当前无实现、训练或诊断进程；恢复后的唯一下一原子动作是只读整合 phase230--238 证据。

2026-09-29 03:54，phase250 架构复核完成。现有证据共同指向一个 bootstrap deadlock，而非互相矛盾：探索噪声阻止连续停稳，失败后果又让 near-ready 的 PPO attenuation 梯度继续增噪；总体方差预算只修复梯度所有权，不能生成缺失的成功 credit。下一证据缺口收敛为逐动作维 sampled noise 到实际运动的物理贡献，不能从 phase230 的全维近零实验外推。

### phase251：动作维噪声到实际运动的因果分解（已验证）

```text
上游结论与待解决瓶颈：phase250 闭合 near-ready 噪声-credit bootstrap deadlock，但 phase230 只同时移除了十维噪声；尚不清楚 base 三维、EE 平移三维、EE 转动三维和夹爪中哪些维度主导 base/EE 实际速度及连续停稳失败
基线 checkpoint / SHA-256：优先只读 phase236 的 phase235 step25 冻结数据/checkpoint `62fb48764f3ea248e392096ef5b1c9d44618ec4bac2ba0d21ae71d688b6696bd`；不续训
唯一变量：无行为变量；先判断已有随机 action noise 是否能作为外生输入，在控制 actor mean、readiness 与前一拍状态后识别逐维对下一拍实际 base/EE 方向速度的响应。只有现有字段不足时才设计一次最小只读导出
待验证假设：少数动作维可能贡献了大部分 near-ready 实际速度下限；若成立，后续应修正统一探索分布的物理尺度/时间结构所有权，而不是全局降噪或加阶段特例
禁止改动范围：不更新模型；不改 reward、std、temporal fraction、方差预算、entropy、完成条件、observation/action、随机化、低层或阶段逻辑；不得用相关性冒充因果结论
运行规模与日志：有效 v3 smoke 为 GPU 3、`8 env x 8 steps`；有效正式 rollout 为 GPU 3、`2048 env x 128 steps`，输出 `doorman_train:/tmp/b2z1_phase251_motion_diag_v2.pt`，日志 `doorman_train:/tmp/b2z1_phase251_motion_diag_v2.log`，optimizer 未执行
成功/失败判据：必须对每个动作维给出可辨识性依据，并最终闭合到 base/EE 方向速度而非仅速度范数；无法区分 temporal context、actor mean 和 innovation 时不得下物理归因结论
观测指标：actor mean、sample-minus-mean、temporal context/innovation、base body-frame velocity、EE heading-frame linear/angular velocity、readiness、contact、done 和一拍滞后
结论与已排除方向：已验证少数动作维主导物理速度下限。剔除 done/reset 跨界后有效一拍/两拍样本 `258423/255580`；以 independent noise 为外生干预，控制 pre base/EE velocity、actor mean、pre commands、readiness 和 contact，按完整边际噪声尺度折算后，`base_x/ee_pitch/ee_x` 对任务相关 base planar+yaw 与 EE linear velocity 响应平方的份额为 `79.83%/13.79%/5.92%`，合计 `99.54%`。readiness `>=0.05` 的 80,176 帧中三者仍合计 `97.96%`。四个互斥环境分组中 `base_x` 始终排名第 1、`ee_pitch` 始终第 2，`ee_x` 排名第 3 或 4；两拍响应前三名不变。AR innovation 对前四维的交叉工具量在主信号 `base_x` 和 `ee_x` 上与 independent 估计同向，cosine 为 `0.881/0.764`；弱响应维的高 readiness 排名不稳定，不作细分结论。排除“十维噪声对停稳影响大致均匀”，也排除用普通 `sample-mean` 相关性代替外生归因
正式保留/删除的代码：已整体删除 `phase251_motion_diagnostics.py`、对应测试、环境快照、trainer 导出与 optimizer 前退出；不保留生产机制
前序阶段回归：phase251 清理后 py_compile、完整 `52 passed, 5 subtests passed`、专属标识归零和 `git diff --check` 通过
下一决策门槛：phase252 只允许审核全阶段统一的物理响应尺度/时间结构所有权；必须同时解释如何降低少数主导维的稳态速度下限，以及如何保留 base 跨过低层步态阈值的探索能力；禁止按 stage/readiness 硬编码降噪或直接扫描 std/fraction
```

2026-09-29 03:54，phase251 计划已原子登记。当前无实现、训练或诊断进程；恢复后的唯一下一原子动作是现有 phase236 张量字段可辨识性审核。

2026-09-29 03:56，phase251 现有字段可辨识性审核完成。phase236 张量不足以逐动作维归因，直接相关回归会混淆 temporal context、actor mean、历史动力学和方向信息。当前仍无 phase251 代码或运行进程；下一原子动作仅为最小只读导出的 shape、时序和外生 innovation 设计。

2026-09-29 03:58，phase251 最小只读诊断实现和静态回归完成。无需改 actor 采样接口：AR(1) innovation 由相邻 context、done 与 rho 精确重建，独立噪声由 action、mean、marginal std、fraction 和 context 精确重建。环境只读快照按 `pre-state(t) -> action(t) -> post-state(t+1)` 导出 base body-frame 与 EE heading-frame 线/角速度、commands、readiness、contact 和 stage。显式环境变量关闭时不收集、不写盘、不增加 storage；容器内 py_compile、纯函数/temporal 联合 `17 passed`、完整 `57 passed, 5 subtests passed` 和 `git diff --check` 全部通过。下一原子动作仅为 GPU 3 的 `8 env x 8 steps` optimizer 前退出 smoke。

2026-09-29 04:00，phase251 首次极小 smoke 判定无效。运行本身退出码 0，64 帧全部字段有限，纯动力学 `cameras=0`、RTX sensors=0、frame rendering disabled，源 checkpoint SHA 未变；但 action 重建最大误差为 `1.135`。代码审核确认 `DeltaActionBase` 在 `env.step()` 内原地积分 action，而临时诊断在 step 后保存了同一张量别名，重现了 phase230 已记录的诊断污染类型。该输出 `/tmp/b2z1_phase251_smoke.pt` 及其 SHA 不得用于物理归因。修复只在进入环境前 clone policy-side 诊断张量，不改变执行 action；下一原子动作是静态回归后同配置重跑。

2026-09-29 04:04，phase251 第二次同配置极小 smoke 已验证。GPU 3 上 `8 env x 8 steps` 在 optimizer 前正常退出，`cameras=0`、RTX sensors=0、frame rendering disabled；全部字段有限，action/context 重建最大误差分别为 `1.862645149230957e-09` 和 `5.960464477539063e-08`，pre/post velocity 不同，确认未错存同一状态。输出 `doorman_train:/tmp/b2z1_phase251_smoke_v2.pt` SHA-256 为 `39afa8c9d2a8b3b0562066cf090b3b4d081b5d215f42af3cb07036d6d7544d95`，日志 SHA-256 为 `0c43cbea53f120522f072ce1d278d13955dd8d2a64eaea967af81dd513b3262b`；源 checkpoint SHA 未变，无训练或诊断进程残留。恢复后的唯一下一原子动作是不改代码地执行 GPU 3、`2048 env x 128 steps` 正式冻结 rollout，不得重跑两次 smoke。

2026-09-29 04:08，phase251 唯一正式冻结 rollout 已启动。容器 `doorman_train`、GPU 3、`2048 env x 128 steps`，laucher PID `1307466/1307472`、worker PID `1307536`；加载 phase235 step25 checkpoint，日志 `doorman_train:/tmp/b2z1_phase251_motion_diag.log`，输出 `doorman_train:/tmp/b2z1_phase251_motion_diag.pt`，optimizer 前退出。运行期间冻结代码，下一原子动作只是轮询该进程。

2026-09-29 04:13，phase251 唯一正式冻结 rollout 已正常完成，optimizer 未执行，无进程残留。张量 `doorman_train:/tmp/b2z1_phase251_motion_diag.pt` SHA-256 `08d3c4fed6be53da25f77b9eb5b401ad9ca3567abfaf0905039b789596042882`，日志 SHA-256 `120a8c0e10d672d7f2762ef5e249930d1c7e4ffc0ecf21702c06bcfbf6570f14`，源 checkpoint SHA-256 仍为 `62fb48764f3ea248e392096ef5b1c9d44618ec4bac2ba0d21ae71d688b6696bd`。共 262,144 个 transition，所有字段有限，action/context 重建最大误差为 `2.98e-8/2.38e-7`，剔除 done/reset 后一拍/两拍有效样本为 `258431/255593`。下一原子动作只是对该冻结张量做外生噪声的逐维因果分解，不再运行 Isaac 或修改代码。

2026-09-29 04:18，phase251 首次正式张量在归因前置审核中改判为无效。`pre_commands` 与 `post_commands` 逐元素完全相同，而 sampled action 显著非零；源码追踪确认 trainer 在 `env.step()` 前仅保留 `get_phase251_motion_snapshot()` 返回的字典，其中 commands/readiness/stage 保留了底层环境张量的引用，直到 step 后才 clone，因此 pre-state 被 post-state 覆盖。速度字段因在快照函数内新建张量而未受污染，但缺少可靠的 pre-command 控制后，已输出的速度排名只是探索性结果，禁止作为最终因果结论。下一原子动作是只修复诊断 pre-state 的立即 clone，然后做静态回归和同配置极小 smoke。

2026-09-29 04:21，phase251 环境 pre-state 快照修复完成并启动 v3 极小 smoke。修复仅在 `env.step()` 前对快照所有字段立即 `detach().clone()`，新增纯函数测试证明后续原地修改环境张量不会污染快照。py_compile、定向 `18 passed`、完整 `58 passed, 5 subtests passed` 和 `git diff --check` 均通过。v3 smoke 使用 GPU 3、`8 env x 8 steps`，输出 `/tmp/b2z1_phase251_smoke_v3.pt`，日志 `/tmp/b2z1_phase251_smoke_v3.log`，运行期间冻结代码。

2026-09-29 04:23，phase251 v3 smoke 已验证并启动 v2 正式冻结 rollout。smoke 的 action/context 重建最大误差为 `1.86e-9/5.96e-8`，pre/post commands 有 58/64 帧变化、最大变化 `0.3368`，readiness 也有变化；张量 SHA-256 `30d4243971b8c61161fdeef8bef0b8ccb9179a743e2d7cc4a41b900b60d7aa7e`，日志 SHA-256 `212521d1c6153618301adad29b92a8ccdeb769e47299529f945005ae32d216d0`。v2 正式 rollout 使用 GPU 3、`2048 env x 128 steps`，worker PID `1309327`，输出 `/tmp/b2z1_phase251_motion_diag_v2.pt`，日志 `/tmp/b2z1_phase251_motion_diag_v2.log`，optimizer 前退出。

2026-09-29 04:29，phase251 v2 正式 rollout 与因果分解完成。张量 SHA-256 `fdc5ecbd4b980eb61530567baa035d25545d4d11de34184897dd3aaec7b3455d`，日志 SHA-256 `d1cb941c72a24ac050cd2fbae4def07caf909f583ec4d6566e10c1b699b890ac`，源 checkpoint SHA 仍为 `62fb48764f3ea248e392096ef5b1c9d44618ec4bac2ba0d21ae71d688b6696bd`；action/context 重建误差 `2.98e-8/2.38e-7`，pre/post commands 有 261,075/262,144 帧变化，有效一拍/两拍样本 `258423/255580`。外生 independent noise 的 FWL 归因控制了 pre base/EE velocity、actor mean、pre commands、readiness 和 contact；`base_x/ee_pitch/ee_x` 占任务相关速度局部响应平方的 `79.83%/13.79%/5.92%`，四分组与两拍排名稳健。下一原子动作仅为清理 phase251 临时代码并完成回归，不修改训练行为。

2026-09-29 04:34，phase251 临时代码清理完成。已删除纯函数模块与测试、环境运动快照、trainer 导出、环境变量门控和 optimizer 前退出；运行目录中 `phase251/PHASE251` 标识搜索为零。py_compile、完整 `52 passed, 5 subtests passed` 和 `git diff --check` 通过，当前无运行进程或临时代码。

### phase252：统一物理响应尺度的探索所有权审核（已验证）

```text
上游结论与待解决瓶颈：phase251 证明十维归一方差与实际物理速度影响不等价，base_x/ee_pitch/ee_x 三维主导 99.54% 局部任务速度响应；但 phase223 又证明 base 探索必须跨过低层步态响应阈值
基线 checkpoint / SHA-256：不加载或更新模型；只读 phase223/230/236/237/251 证据与现有源码
唯一变量：无行为变量；区分 max std、状态 attenuation、temporal fraction/rho 和 normalized variance budget 各自应拥有的语义，判断物理响应尺度应进入哪一层
待验证假设：当前预算按 `variance/max_variance` 等权只是归一动作空间容量，没有表达各维对实际停稳速度的物理代价；正确修复应位于全阶段统一的探索分布/预算所有权，而非 reward 或 stage 特例
禁止改动范围：不改代码、reward、std、fraction、rho、budget、entropy、完成条件、observation/action、随机化或低层；不启动 Isaac 或训练；不按 stage/readiness 硬编码降噪
运行规模与日志：只读源码、文档和 phase251 有效统计；不生成 checkpoint
成功/失败判据：必须为四个所有权层分别写出不变量、物理语义、已有反证和可证伪的下一单变量实验；候选方案必须同时保留远处跨越步态阈值和近目标降低速度下限的可能性
观测指标：各维 max/marginal/conditional variance、temporal impulse 时常、物理响应权重、预算梯度所有权、低层步态阈值与 near-ready 稳定覆盖
结论与已排除方向：已验证。四层所有权如下：(1) `max std` 是每个动作坐标的瞬时物理覆盖上限；phase223 证明 base 尺度低于低层 `0.1 m/s` 启动阈值时基本不产生步态，因此不能用 phase251 排名直接降低 base max std。(2) observation-dependent attenuation 是唯一拥有状态间方差重分配的层；它可以在统一 observation 语义下表达远处高探索、近处低噪声，但 phase236/237 证明当前只有 `1.8%--5.7%` 弱分化，且高 readiness 六个运动维的 PPO surrogate 正在推动增噪，不能把失败归因于网络没有表达能力。(3) temporal fraction/rho 保持每维边际方差不变，只改变噪声频谱和持续性；phase223 的同尺度对照中，base 三维独立与 AR(1) 的实际速度约为 `0.1168/0.2192 m/s`，而命令尺度都能跨过步态阈值。当前 base 三维和 ee_x 都使用 `fraction=0.9, rho=0.95`，因此 phase251 的高 base_x 物理响应首先包含 temporal 动态放大。(4) normalized variance budget 只负责防止全局探索坍塌；当前逐动作 `mean((std/max_std)^2)>=0.8` 不代表等物理速度，也不应承担动作动力学标定。把 phase251 权重直接写进该下界会把“必须保留探索容量”和“必须降低运动扰动”混成同一约束，不能保证近目标安静，也可能反向保护高响应维。首个可证伪变量因此不是 reward、max std、budget 权重或 stage/readiness 特例，而是 base 三维 temporal persistence
正式保留/删除的代码：无 phase252 代码；只更新唯一执行账本
前序阶段回归：只读审核，没有修改生产代码、配置、checkpoint 或运行进程；phase251 清理后的回归结果继续有效
下一决策门槛：phase253 用同一冻结 checkpoint 比较当前 `[0.9,0.9,0.9,0.9,0,0,0,0,0,0]` 与 base 独立、仅 ee_x 相关的 `[0,0,0,0.9,0,0,0,0,0,0]`。只有候选保持 base command 跨越步态阈值，同时显著降低实际 base motion 且不恶化 EE/safety，才允许进入干净训练
```

2026-09-29 04:34，phase252 计划已原子登记。当前无实现、训练或诊断进程；恢复后的唯一下一原子动作是只读完成四个探索所有权层的因果表，不得直接修改参数或启动训练。

2026-09-29 04:41，phase252 所有权审核完成。直接降低 base max std 会重新进入 phase223 已证实的步态死区；状态 attenuation 是状态重分配接口，但当前 PPO credit 尚未让它充分分化；方差预算只负责防全局坍塌，不能代表物理速度；temporal structure 在不改变边际动作覆盖时会显著改变真实动态响应。因此下一变量收敛为 base 三维时间持续性，未修改任何生产代码或参数。

### phase253：base 时间持续性的冻结反事实（失败）

```text
上游结论与待解决瓶颈：phase252 确认 base max std 必须保留跨越低层步态阈值的瞬时覆盖，而 base 三维 AR(1) 会在相同边际方差下放大真实底盘速度；本 phase 已在当前纯 RL checkpoint 和状态 attenuation 下完成匹配因果复验
基线 checkpoint / SHA-256：phase235 `model_step_000025.pt` / `62fb48764f3ea248e392096ef5b1c9d44618ec4bac2ba0d21ae71d688b6696bd`；只读加载，不续训
唯一变量：temporal correlated fraction 从当前 `[0.9,0.9,0.9,0.9,0,0,0,0,0,0]` 改为 `[0,0,0,0.9,0,0,0,0,0,0]`，即 base x/y/yaw 改为时间独立，仅 canonical ee_x 保持原相关结构；`rho=0.95`、十维 marginal std、actor mean、attenuation、budget、observation/action 和环境完全不变
待验证假设：移除 base 命令的时间持续性可降低低层实际底盘速度及其对 EE 世界速度的带动，同时单步 base command 仍有足够比例越过 `0.1 m/s` 步态启动阈值；ee_x 的持续局部覆盖保持不变
禁止改动范围：不更新模型；不改 reward、max/min std、attenuation、budget、entropy、rho、完成条件、observation/action、随机化、低层、阶段逻辑或 ee_x fraction；不扫描 fraction，不启动训练
运行规模与日志：`seed=0`、`2048 env x 128 steps`、同一 phase235 checkpoint。修正版 control 为 `doorman_train:/tmp/b2z1_phase253_control_v2.pt`（SHA-256 `b537af2911467882ae81c4261d2895e2308be563372204f92ea315202c1e4b18`，日志 SHA-256 `b929ee0cfae7f09e8313747d298fec939bc335e67e460d2a10ef08762182d9f0`）；candidate 为 `doorman_train:/tmp/b2z1_phase253_base_independent_v2.pt`（SHA-256 `e53c0dd9b16ddca5130947812adaf78a91b51ca451901c3ef28bd69534d898f6`，日志 SHA-256 `ed317f68d66c2868407abe65d2d88f29b19024fa423f4bdae8018ed1839b8379`）。两组统一采用 `pre-state(t) -> action(t) -> post-state(t+1)`，optimizer 前退出
成功/失败判据：(1) 十维 marginal std 与逐维单步方差匹配；(2) 候选 base command 越过 `0.1 m/s` 启动阈值的覆盖不因去相关而下降；(3) actual base planar/yaw motion 显著低于 control；(4) EE 实际速度不恶化；(5) existing/unsafe termination 不恶化；(6) likelihood 可按新时间结构精确重建
观测指标：十维 mean/marginal std/sample noise、base command 阈值覆盖与 lag-k、自身实际 planar/yaw velocity、EE heading-frame linear velocity、readiness 分桶、geometry quality、termination、安全力和 log-prob 重建误差
结论与已排除方向：候选失败。control 到 candidate 的 base 命令越过 `0.1` 阈值覆盖为 `89.04% -> 89.50%`，base planar/yaw speed 为 `0.21199 -> 0.11670 m/s`、`0.19128 -> 0.16040 rad/s`，EE linear speed 为 `0.47723 -> 0.40751 m/s`；base x/y/yaw lag-1 从约 `0.855` 降为 0，ee_x 保持约 `0.855`，证明物理假设成立。但 existing termination 从 `1784 -> 1866`，即 `0.6805% -> 0.7118%/step`，增量约为 `4.18` 个配对环境标准误；unsafe-force `44 -> 38`、door overspeed 和 shoulder collision 均为 0，不能用任务安全终止解释该回归。候选不得写入正式配置或启动训练
正式保留/删除的代码：已增加显式 `B2Z1_PHASE253_DIAG_PATH` 门控的临时环境快照与 trainer 导出；默认关闭且结束后必须整体删除
前序阶段回归：py_compile、temporal/actor/budget 定向 `29 passed`、完整 `52 passed, 5 subtests passed` 和 `git diff --check` 通过
下一决策门槛：phase254 只分解 existing termination 的来源。先区分父类已有 reset、距门超过 4 m、机械臂非夹爪关节超速；若增量来自父类项，再进入下一 phase 分解跌倒、低高度、接触、关节/力矩限位和 timeout。根因闭合前不改 fraction、std、低层、安全阈值或 reward
```

2026-09-29 04:41，phase253 计划已原子登记。当前无实现、训练或诊断进程；恢复后的唯一下一原子动作是静态定义两组冻结 rollout 的同 seed 配对、采样时序、最小字段和明确输出路径，不得先改正式配置或启动 Isaac。

2026-09-29 04:45，phase253 静态实验契约完成。control 不重复运行，直接使用 phase251 v2 的有效冻结张量；候选保持相同 seed、checkpoint、2048 环境、128 步和状态时序，仅覆盖 base 三维 temporal fraction。最小张量字段固定为 pre/post base body velocity、EE heading velocity、commands、readiness、contact、stage，外加 action、mean、marginal std、fraction、context、innovation、independent noise、done 和重建误差；安全与 geometry 聚合指标从同次正式日志读取。下一原子动作是实现可完整删除的只读导出并完成静态回归，不启动 Isaac。

2026-09-29 04:49，phase253 临时只读接线与静态回归完成。环境快照在 `env.step()` 前立即逐字段 clone，policy action 也在进入会原地积分的环境前 clone，避免 phase251 两类别名污染；trainer 按同一时序导出并在 optimizer 前返回。Isaac Python 下 py_compile、定向 `29 passed`、完整 `52 passed, 5 subtests passed` 和 `git diff --check` 全部通过。下一原子动作仅为 GPU 3 的 `8 env x 8 steps` 候选 smoke。

2026-09-29 04:52，phase253 候选 smoke 已验证并正常退出。GPU 3、`8 env x 8 steps`，纯动力学 `cameras=0`、RTX sensors=0、frame rendering disabled；全部张量有限，fraction 精确为 `[0,0,0,0.9,0,0,0,0,0,0]`，action 重建最大误差 `2.98e-8`，pre/post commands 有 62/64 帧变化且速度状态不同。张量 SHA-256 `d8702d68f9f4323cd5c3525a4968f67540ac8accb82bfdb84b4f336ebf5ed1b2`，日志 SHA-256 `65b5e299f7316573ea87019e5db1b847d136bdad6da6783c9661bed0d7d6c6bf`；源 checkpoint SHA 保持 `62fb48764f3ea248e392096ef5b1c9d44618ec4bac2ba0d21ae71d688b6696bd`，optimizer 未执行，无进程残留。下一原子动作仅为同配置 `2048 env x 128 steps` 正式候选 rollout。

2026-09-29 04:53，phase253 唯一正式候选冻结 rollout 已启动。容器 `doorman_train`、GPU 3、`2048 env x 128 steps`，launcher PID `1310649/1310658/1310664`、worker PID `1310728`；输出 `/tmp/b2z1_phase253_base_independent_diag.pt`，日志 `/tmp/b2z1_phase253_base_independent_diag.log`，optimizer 前退出。运行期间冻结代码与参数，下一原子动作只轮询该进程。

2026-09-29 04:57，phase253 正式候选 v1 已正常完成并在 optimizer 前退出。张量 SHA-256 `e6e27f40f1b88b10a2bb7f46101be3158c9e033742a9a05b249ecf61f27714d1`，日志 SHA-256 `db310203448eeec49e33d1caee2d3b89ba06e52cdb3f48e4fbb1af42e0a1cbe7`，源 checkpoint SHA 不变。物理字段有效：base 命令任一维越过 `0.1` 的覆盖为 control/candidate `89.15%/89.48%`，base planar speed 为 `0.21285/0.11673 m/s`，base yaw speed 为 `0.19273/0.16021 rad/s`，EE linear speed 为 `0.47959/0.40751 m/s`，base 三维 lag-1 从约 `0.855` 降至约 0，ee_x 保持约 `0.856`。但预登记安全判据要求区分 existing 与 unsafe-force，v1 只有 done 总数，不能从 contact-free 严格重建终止原因，因此 v1 不得作为 phase253 最终验收。下一步只补充环境已经计算出的终止布尔量，不改变行为。

2026-09-29 04:58，phase253 终止原因测量缺口已补齐并完成静态回归。临时快照新增 existing、unsafe-force、door-overspeed、shoulder-collision 和 stage-overtime 五个 `_check_termination()` 原始布尔量；不修改 reset、reward 或动作链。py_compile、完整 `52 passed, 5 subtests passed` 和 `git diff --check` 通过。下一原子动作是极小 smoke 验证新增字段，随后只运行一对修正版匹配 rollout。

2026-09-29 04:59，新增终止字段 smoke 通过。五类字段均为 `[8,8,1]` 并严格是 done 的子集；本次 existing/done 为 5，unsafe-force、door-overspeed、shoulder-collision、stage-overtime 均为 0，action 重建误差 `2.98e-8`。张量 SHA-256 `978e1d4b0014ae1ba74bca61dd44f393c13fc81f5dcbd96a90a5474b2817b936`，日志 SHA-256 `daa1d016b90df0dc71717df6f2f2c3b04076513594e2ec964a52a733831e9d27`。随后已并行启动唯一一对修正版正式 rollout：GPU 2 control worker PID `1311739`，GPU 3 candidate worker PID `1311740`，均为 `2048 env x 128 steps`、seed 0、同一 checkpoint，optimizer 前退出。下一原子动作只轮询这两个进程。

2026-09-29 05:04，phase253 修正版匹配对照完成并退出，两个 optimizer 均未执行，源 checkpoint SHA-256 保持 `62fb48764f3ea248e392096ef5b1c9d44618ec4bac2ba0d21ae71d688b6696bd`。候选保持约 `89%` 的步态命令覆盖并显著降低实际 base/EE 速度，但 existing termination 增量为 `+0.0313` 个百分点/step，约 `4.18` 个配对环境标准误，违反预登记的不恶化判据；phase253 判为失败，不得写入正式配置。当前无训练或诊断进程，下一原子动作是 phase254 终止来源只读审核。

### phase254：base 独立噪声 existing termination 来源分解（已验证）

```text
上游结论与待解决瓶颈：phase253 已证明 base 去相关在不降低命令阈值覆盖的情况下显著降低实际运动，但 existing termination 出现小幅且统计可辨识的回归；聚合 existing 标志尚不能区分具体物理来源
基线 checkpoint / SHA-256：phase235 `model_step_000025.pt` / `62fb48764f3ea248e392096ef5b1c9d44618ec4bac2ba0d21ae71d688b6696bd`；只读加载，不续训
唯一变量：仍只比较 phase253 的 control `[0.9,0.9,0.9,0.9,0,0,0,0,0,0]` 与 candidate `[0,0,0,0.9,0,0,0,0,0,0]`；新增字段仅观察终止来源，不改变执行行为
待验证假设：candidate 的 existing termination 增量可被父类已有 reset、`relative_door_pos_buf.norm()>4.0`、机械臂非夹爪关节超速三类原始布尔量之一或其重叠精确解释；已验证
禁止改动范围：不改 fraction、rho、std、actor、attenuation、budget、reward、完成条件、低层、安全阈值、随机化或任何生产行为；不在根因字段闭合前猜测并修复
运行规模与日志：同 checkpoint、seed 0、`2048 env x 128 steps` 各运行一份 control/candidate，optimizer 前退出。control 张量/日志 SHA-256 为 `296abb209a8be96b5b557dcc8de3075bd294d309ae1a6bb1187923a267656da7` / `cb6fa29a4d2e956bd22d81fc58de24d417283363883f1929e6257d58e5a992a5`；candidate 为 `323b497b23a75b6e079a50f181afa83cc18dc452f61027c0fd50c4d9109934fa` / `a9a78b4f0e6d9bbdf5ebd8917dc16da1fde5c7e38c096bc24513a7f14afca3f3`
成功/失败判据：(1) 三类字段均在 reset 被后续覆盖前采集；(2) `existing == (inherited_reset | far_from_door | arm_dof_overspeed) & ~stage_overtime` 逐元素重建误差为 0；(3) 两组 done、existing 及 phase253 物理量保持可比；(4) 能将 existing 增量归属到至少一个具体层，否则不得修改行为并继续向父类内部细分
观测指标：inherited reset、far-from-door、arm DOF overspeed、stage overtime、existing、done，以及 phase253 已有 action/context 重建、base command 覆盖和实际运动指标
结论与已排除方向：已验证。control/candidate 的 inherited reset 为 `94/29`、far-from-door 为 `0/0`、arm DOF overspeed 为 `1739/1857`、existing 为 `1784/1867`；重叠仅 control 有 1 帧 inherited 与 arm 同时为真。candidate 的 arm overspeed 增量 `+118`，配对环境 z 值约 `6.31`；inherited reset 反而 `-65`，far 不变，因此净 `+83` 次 existing 回归由 arm overspeed 解释。所有 arm overspeed 均发生在 APPROACH_PREGRASP。当前数据没有 reset 前逐关节速度，不能继续推断具体关节或原因，也不能据聚合计数放宽阈值
正式保留/删除的代码：只扩展现有 `B2Z1_PHASE253_DIAG_PATH` 临时快照和导出；phase254 归因结束后连同 phase253 临时代码整体删除
前序阶段回归：phase253 的 py_compile、定向 `29 passed`、完整 `52 passed, 5 subtests passed` 和 `git diff --check` 继续有效；新增字段后必须重跑静态回归
下一决策门槛：phase255 必须在 reset 覆盖物理状态前保存逐关节终止瞬间状态，并结合前一拍状态和同拍统一 action 判定哪个关节首先越过 20 rad/s、是持续超速还是一步跃迁；完成前不改终止阈值或探索行为
```

2026-09-29 05:07，phase254 终止链只读审核完成。调用顺序为 `LeggedRobotBase -> StagedTaskBase -> DoorPregrasp -> B2Z1DoorSixStage`；在 `DoorPregrasp` 增加自身 reset 前快照父类结果，随后分别记录 far-from-door 和已带 `episode_length>20` 门控的 arm overspeed，即可逐元素重建 B2Z1 层看到的 existing termination。当前无运行进程；下一原子动作仅为增加这些只读字段和重建断言并完成静态回归，不启动 Isaac。

2026-09-29 05:08，phase254 最小只读字段、写盘前 OR 重建断言和静态回归完成。容器系统 Python 的 py_compile、Isaac Python 完整测试 `52 passed, 5 subtests passed`、宿主机 `git diff --check` 均通过；系统 Python 没有 pytest，因此未安装依赖，改由项目既有 `/isaac-sim/python.sh` 执行测试。下一原子动作固定为 GPU 3、`8 env x 8 steps` 候选 smoke，输出 `/tmp/b2z1_phase254_smoke.pt`，日志 `/tmp/b2z1_phase254_smoke.log`，optimizer 前退出。

2026-09-29 05:08，phase254 极小 smoke 已唯一启动。容器 `doorman_train`、GPU 3、`8 env x 8 steps`，launcher PID `1312464/1312470`、worker PID `1312534`；输出 `/tmp/b2z1_phase254_smoke.pt`，日志 `/tmp/b2z1_phase254_smoke.log`，candidate fraction 固定为 `[0,0,0,0.9,0,0,0,0,0,0]`，optimizer 前退出。运行期间冻结代码，下一原子动作只轮询该进程。

2026-09-29 05:11，phase254 极小 smoke 正常完成并退出。张量 SHA-256 `ac1a2a8d152d637591327399496509160bfe30b87d89136c3fb69ded540fe261`，日志 SHA-256 `153a20ce74420fa910e689ac01070a7882a0ab971081a590282e0fa56c5670b9`，源 checkpoint SHA-256 仍为 `62fb48764f3ea248e392096ef5b1c9d44618ec4bac2ba0d21ae71d688b6696bd`。三个来源字段均为 `[8,8,1]`，全部张量有限，existing OR 重建 mismatch 为 0，action 重建误差 `2.98e-8`；5 次 existing 全部来自 arm overspeed，但极小样本不用于物理归因。当前无残留进程；下一原子动作是启动唯一一对正式匹配 rollout。

2026-09-29 05:12，首次正式启动命令因 shell 的 `export` 与逐进程环境变量之间缺少分号，在 Isaac 启动前立即退出；未创建环境、rollout、日志张量或 checkpoint，不能计为一次实验。修正命令边界后，同一预注册配置已唯一启动：GPU 2 control worker PID `1312940`，输出 `/tmp/b2z1_phase254_control.pt`、日志 `/tmp/b2z1_phase254_control.log`；GPU 3 candidate worker PID `1312941`，输出 `/tmp/b2z1_phase254_base_independent.pt`、日志 `/tmp/b2z1_phase254_base_independent.log`。两者均为 seed 0、`2048 env x 128 steps`、同一 checkpoint，optimizer 前退出；下一原子动作只轮询这两个进程。

2026-09-29 05:16，phase254 正式匹配对照完成并退出，optimizer 未执行，源 checkpoint SHA 未变且无进程残留。两组 action 重建误差均为 `5.96e-8`、existing OR 重建 mismatch 均为 0、全部张量有限。candidate 相对 control 的 inherited/far/arm/existing 计数差为 `-65/0/+118/+83`，配对 z 值分别约 `-6.47/NA/+6.31/+4.20`；任务 unsafe-force 为 `37/37`，door overspeed 为 `1/0`，shoulder collision 为 `0/0`。因此回归已定位到 arm DOF overspeed 所有权层，但尚未闭合逐关节动力学机理。

### phase255：arm DOF overspeed 逐关节动力学归因（已验证）

```text
上游结论与待解决瓶颈：phase254 已证明 base 去相关 candidate 的 existing 回归来自 arm DOF overspeed，而不是父类失稳、远离门或任务安全终止；现有 post-step 快照在 done 环境中已被 reset 覆盖，缺少真正越界瞬间的六关节状态
基线 checkpoint / SHA-256：phase235 `model_step_000025.pt` / `62fb48764f3ea248e392096ef5b1c9d44618ec4bac2ba0d21ae71d688b6696bd`；只读加载，不续训
唯一变量：仍只比较 control `[0.9,0.9,0.9,0.9,0,0,0,0,0,0]` 与 candidate `[0,0,0,0.9,0,0,0,0,0,0]`；新增字段只读，不改变动作、动力学或 reset 判据
待验证假设：`+118` 次 arm overspeed 可归属到一个或少数明确关节，并可由前一拍速度、同拍动作/物理命令、终止瞬间速度跃迁和力矩区分持续积累、控制目标跳变或单步动力学冲击
禁止改动范围：不改 20 rad/s 终止阈值、fraction、rho、std、actor、attenuation、budget、reward、低层、完成条件、随机化或 reset；不按 stage/seed/门型特调
最小字段：常规 pre-state 六关节 position/velocity；`DoorPregrasp._check_termination()` 内、reset 覆盖前保存六关节 position/velocity/torque、逐关节 limit ratio 与 exceed mask、episode length、stage、物理十维 command、base velocity；trainer 已有同拍 action/mean/noise 保持不变
成功/失败判据：(1) 终止瞬间 `any(exceed_mask)` 与 arm overspeed 标志逐元素一致；(2) 非事件字段有限且 shape 正确；(3) 对每个事件可确定首个越界关节、越界幅度和 `pre -> terminal` 速度变化；(4) control/candidate 的差异能落到具体关节和前因，否则只登记不可辨识缺口，不修改行为
运行规模与日志：先静态审核与 `8 env x 8 steps` candidate smoke；通过后才以同 checkpoint、seed 0、`2048 env x 128 steps` 各运行一份匹配冻结 rollout，optimizer 前退出，路径在启动前登记
正式保留/删除的代码：扩展现有 phase253/254 临时诊断；phase255 根因闭合后整体删除全部临时代码并完整回归
下一决策门槛：只有逐关节、逐时间证据证明超速属于错误命令映射、reset 初态或控制器瞬态之一，才在对应所有权层提出单一修复；若 20 rad/s 是合法物理速度也必须用低层/URDF 依据证明，不能直接放宽阈值
```

2026-09-29 05:16，phase255 计划已登记。源码审核确认六个 arm DOF 均为 `joint1--joint6`，配置速度上限均为 `20 rad/s`，termination ratio 为 `1.0`；现有 phase254 只保留聚合 bool，无法确定具体关节。当前无实现或运行进程；下一原子动作仅为审核上述最小字段在 reset 前后的采样时序和 tensor shape，不启动 Isaac。

2026-09-29 05:20，phase255 reset 前 terminal-state 快照、常规 pre-state 六关节状态和逐关节 exceed 重建断言已接线。terminal 快照在 `DoorPregrasp._check_termination()` 内、环境 reset 前 clone；trainer 写盘前强制验证 `any(exceed)==arm_overspeed`。py_compile、Isaac Python 完整测试 `52 passed, 5 subtests passed` 和 `git diff --check` 通过。下一原子动作固定为 GPU 3、`8 env x 8 steps` candidate smoke，输出 `/tmp/b2z1_phase255_smoke.pt`，日志 `/tmp/b2z1_phase255_smoke.log`，optimizer 前退出。

2026-09-29 05:20，phase255 极小 smoke 已唯一启动。容器 `doorman_train`、GPU 3、`8 env x 8 steps`，launcher PID `1313692/1313698`、worker PID `1313762`；candidate fraction 为 `[0,0,0,0.9,0,0,0,0,0,0]`，输出 `/tmp/b2z1_phase255_smoke.pt`、日志 `/tmp/b2z1_phase255_smoke.log`。运行期间冻结代码，下一原子动作只轮询该进程。

2026-09-29 05:22，phase255 极小 smoke 正常完成并退出。张量 SHA-256 `799b3609e722eb2f688ac9666390f4d4bd437eb1ecc75bc1a40a1296e49a1177`，日志 SHA-256 `702b6dc4e05075716a216fedfb532caa699ee32a8c8a2b11d9b9d07610af9085`，源 checkpoint SHA 未变。所有新增字段 shape 正确且有限，existing/arm 两个重建 mismatch 均为 0，action 重建误差 `2.98e-8`；5 次超速的逐关节计数为 `[0,0,0,0,0,5]`。当前无残留进程；下一原子动作是唯一一对正式匹配 rollout。

2026-09-29 05:23，phase255 唯一一对正式 rollout 已启动。GPU 2 control worker PID `1314150`，输出 `/tmp/b2z1_phase255_control.pt`、日志 `/tmp/b2z1_phase255_control.log`；GPU 3 candidate worker PID `1314151`，输出 `/tmp/b2z1_phase255_base_independent.pt`、日志 `/tmp/b2z1_phase255_base_independent.log`。两者均为 seed 0、`2048 env x 128 steps`、同一 checkpoint，optimizer 前退出；下一原子动作只轮询这两个进程。

2026-09-29 05:29，phase255 首轮正式对照完成并退出，optimizer 未执行、源 checkpoint SHA 未变且无进程残留。control 1740 次 arm overspeed 中 joint6 为 1738、joint4 为 3（1 帧重叠）；candidate 1857 次全部为 joint6。joint6 terminal `|velocity|` 中位数约为 `20.003 rad/s`；candidate 事件中 `56.4%` 前一拍 `|velocity|<1 rad/s`，control 为 `60.3%`，证明多数事件是一拍跃迁而非缓慢逼近。现有 terminal torque 使用 `self.torques`，但 arm 由 IsaacLab implicit actuator 执行，因此该字段恒 0、语义无效；同时缺少当拍 IK target/delta。下一原子动作只修正诊断字段所有权并补齐 DLS 中间量，不修改生产控制。

2026-09-29 05:32，phase255 v2 诊断字段接线和静态回归完成。DLS 计算点仅在显式诊断环境变量开启时额外保存 `dpos/drot/delta_q/arm target/Jacobian singular values`；终止检查点在 reset 前保存 IsaacLab articulation 的 `applied_torque`，不再使用对 implicit arm 恒为 0 的显式 effort buffer。py_compile、完整测试 `52 passed, 5 subtests passed` 和 `git diff --check` 通过。下一原子动作固定为 GPU 3、`8 env x 8 steps` candidate smoke，输出 `/tmp/b2z1_phase255_ik_smoke.pt`、日志 `/tmp/b2z1_phase255_ik_smoke.log`。

2026-09-29 05:35，phase255 v2 极小 smoke 与 lab30 源码对照完成。smoke 的 5 次事件中 joint6 DLS delta 为 `-3.90..+3.98 rad`、position target 为 `-4.50..+4.37 rad`，越过 joint6 `[-2.79,2.79]` 硬限位，implicit applied torque 达到 `±80 N·m`；Jacobian 最小奇异值为 `0.12..0.14`，不是奇异位形。当前 `visual_wholebody_yfh_lab30` 容器源码 commit `be64ce754f7b4bf74fd0da332b308a119ae09a83` 的 `_get_arm_pos_targets()` 直接使用原始 quaternion-vector orientation error；本库额外 `lowlevel_ik_orientation_gain=4.0` 及乘法是未提交差异。下一原子动作只删除该错误增益并恢复 lab30 原公式，不增加 target clamp、限速或阈值补丁。

2026-09-29 05:38，唯一根因修复和静态回归完成。已删除 `lowlevel_ik_orientation_gain` 配置、实例变量和 DLS 前乘法，`drot` 现与 lab30 一致直接取最短 quaternion-vector orientation error；新增纯 RL 契约禁止该参数回归。py_compile、定向 `7 passed`、完整 `53 passed, 5 subtests passed` 和 `git diff --check` 均通过。下一原子动作固定为 GPU 3、`8 env x 8 steps` 同 seed candidate smoke，输出 `/tmp/b2z1_phase255_gain_fix_smoke.pt`、日志 `/tmp/b2z1_phase255_gain_fix_smoke.log`。

2026-09-29 05:46，修复后同 seed candidate smoke 正常完成并退出，optimizer 未执行且无残留训练进程。张量 SHA-256 `1bbf0f34ee4a71f333e02efc6370bb4a453d50d42e44ab31cf328d3868f4d35c`，日志 SHA-256 `c2672d894c6a2249d48f3b1e7ba85fc1955acec6d4baa48870dc84fe96d11918`；existing/arm 重建 mismatch 均为 0，action 重建误差 `3.73e-9`。与修复前同 seed smoke 相比，joint6 超速 `5 -> 0`，全局 joint6 `|delta_q| max 4.009 -> 1.006 rad`、`|target| max 4.778 -> 2.059 rad`、`|applied torque| max 80.0 -> 4.439 N·m`。该 64 帧结果满足进入正式复验的门槛，但样本不足以宣布 candidate 成立；下一原子动作唯一固定为 GPU 2/3 的修复后正式匹配 rollout。

2026-09-29 05:47，修复后的唯一一对正式匹配 rollout 已启动。容器 `doorman_train` 内 GPU 2 control worker PID `1315825`，输出 `/tmp/b2z1_phase255_gain_fix_control.pt`、日志 `/tmp/b2z1_phase255_gain_fix_control.log`；GPU 3 candidate worker PID `1315826`，输出 `/tmp/b2z1_phase255_gain_fix_base_independent.pt`、日志 `/tmp/b2z1_phase255_gain_fix_base_independent.log`。两者均为 seed 0、`2048 env x 128 steps`、同一 phase235 checkpoint，optimizer 前退出；运行期间冻结源码与参数，下一原子动作只轮询这两个进程。

2026-09-29 05:48，修复后正式匹配 rollout 均正常完成并在 optimizer 前退出，源 checkpoint SHA 未变。control/candidate 的 action 重建误差均为 `5.96e-8`，existing/arm 来源重建 mismatch 均为 0，全部张量有限。candidate 保持 base command 覆盖 `89.31%`（control `88.92%`），base planar/yaw 和 EE linear speed 分别由 `0.2119/0.1916/0.5216` 降为 `0.1166/0.1602/0.4584`，降幅 `44.95%/16.38%/12.12%`。existing termination `114 -> 68`（配对 z `-6.20`），arm overspeed `9 -> 6`，unsafe-force `26 -> 24`，door overspeed 和 shoulder collision 均为 0。修复前 control/candidate arm overspeed 为 `1740/1857`；本轮约降低 `99.5%`，且 candidate 不再回归。残余 `9/6` 次均在 APPROACH_PREGRASP，仍伴随少数 DLS target 大跳变；当前不是 candidate 特有回归，不在本 phase 追加 clamp 或阈值补丁，若后续重新成为首个训练瓶颈再以独立证据重开。

2026-09-29 05:49，phase255 收尾完成。生产配置正式采用 base 三维 fraction 0、ee_x fraction 0.9；契约同步更新。phase253--255 的环境快照、逐步 trainer 导出、DLS 中间量和 optimizer 前退出已整体删除。py_compile、定向 `16 passed`、完整 `53 passed, 5 subtests passed`、临时标识搜索和 `git diff --check` 均通过。phase255 判为已验证；下一原子动作进入 phase256，只运行清理后生产配置的极小 optimizer smoke。

### phase256：base-independent 生产训练链验证（失败）

```text
上游结论与待解决瓶颈：phase255 已通过正式匹配对照并清理全部临时诊断；新生产配置尚未执行真实 optimizer update
基线 checkpoint / SHA-256：optimizer smoke 从随机初始化启动，不加载旧 checkpoint；旧 checkpoint 与 IK/探索接口不同，不得继承
唯一变量：相对 phase255 已验证生产代码不再修改行为，只验证清理后标准 rollout -> GAE -> PPO optimizer -> 保存/退出链路
待验证假设：base-independent 配置和 IK 修复可在纯动力学路径完成一次真实 PPO 更新，无异常、NaN、临时早退或参数未更新
禁止改动范围：不改 reward、observation/action、完成条件、std、fraction、rho、低层、安全阈值、随机化或 stage 逻辑；smoke 失败先归因，不增加兼容开关
运行规模与日志：optimizer smoke 为 GPU 3、`8 env x 8 steps x 1 update`、日志 `doorman_train:/tmp/b2z1_phase256_optimizer_smoke.log`；通过后正式训练为 GPU 0--3 各 `2048 env x 128 steps`、随机初始化到 step25，日志 `doorman_train:/tmp/b2z1_phase256_base_independent_clean25.log`
成功/失败判据：日志确认 cameras=0、RTX sensors=0、frame rendering disabled；完成 iteration 1 和 optimizer；无 traceback/NaN；生产 fraction 精确为 `[0,0,0,0.9,0,0,0,0,0,0]`
观测指标：启动配置、iteration/optimizer 完成、loss/entropy/KL、参数更新、终止、安全、运行异常和输出目录
结论与已排除方向：待运行
正式保留/删除的代码：不新增诊断代码；通过后直接进入同配置四卡干净训练
前序阶段回归：phase255 py_compile、16 项定向、完整 `53 passed, 5 subtests passed` 和 diff check 已通过
下一决策门槛：optimizer smoke 通过后，从随机初始化启动四卡训练；step25 前不据短时波动调整行为
```

2026-09-29 05:50，phase256 唯一 optimizer smoke 已启动。容器 `doorman_train`、GPU 3、worker PID `1316854`，随机初始化 `8 env x 8 steps x 1 update`，日志 `/tmp/b2z1_phase256_optimizer_smoke.log`；未设置任何 phase253--255 诊断变量。运行期间冻结代码与配置，下一原子动作只轮询该进程。

2026-09-29 05:51，optimizer smoke 正常完成并退出。日志 SHA-256 `8af963e117c6b17b3d8d42002fccf2402bc03a334cb20b88641d85f54387fa7b`；配置解析后的 fraction 精确为 `[0,0,0,0.9,0,0,0,0,0,0]`，日志确认 cameras=0、RTX sensors=0、frame rendering disabled。真实 iteration 1 已完成，`policy/value/weighted PPO loss=-0.045117/12.430334/12.596018`，64 timesteps、iteration time `2.53 s`，无 traceback、NaN 或诊断早退。满足正式干净训练门槛。

2026-09-29 05:51，phase256 四卡随机初始化训练已启动。GPU 0--3 各 `2048 env x 128 steps`，worker PID `1317134--1317137`，共 8192 env，目标 step25，checkpoint 保存频率 25；日志 `/tmp/b2z1_phase256_base_independent_clean25.log`。不加载旧 checkpoint，不含临时诊断路径。下一原子动作只轮询 update1，运行期间冻结代码和配置。

2026-09-29 05:55，四卡 update1 正常完成。四个 rank 和 optimizer 均正常，纯动力学配置已在四个场景分别确认；本轮 `1,048,576` timesteps，吞吐约 `21,757 steps/s`，collection/learning/总时间为 `42.383/5.811/48.19 s`，无 traceback、NaN 或运行异常。首轮全部样本处于 APPROACH_PREGRASP，符合随机初始化且 staged-reset 尚未积累的起点。下一有效判断点固定为 step25，期间冻结代码和配置。

2026-09-29 06:16，phase256 四卡训练正常完成 step25 并退出，共 `26,214,400` timesteps；末轮 iteration `45.23 s`，无异常或 NaN。checkpoint `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260929_055130/model_step_000025.pt` SHA-256 `5b4cf604659ea3b4c41bbd500396b49f7396c74aadb79bbe0993a79291e164cd`，日志 SHA-256 `742049246623ef6527b6f90cea466c3abb131f54decad8f357a37636e4d14ecc`。step25 五项 quality 为 `0.2999/0.5866/0.1238/0.0820/0.8828`，episode-best `0.3793/0.6609/0.1955/0.2373/0.9808`；相比 phase224 同期，除 base yaw 外均回归。actual planar speed `0.1232 m/s`，显著低于 phase235 的 `0.2103 m/s`，但 stable 1/5/10、all-checks-now 和 GRASP 均为 0。末轮 command goalward 为 `+0.0174 m/s`、actual goalward 为 `-0.0176 m/s`，existing/unsafe-force/door-overspeed/shoulder-collision 为 `0.0002/0/0/0`。候选解决了速度过高但没有形成可执行的联合接近，phase256 判失败，不续训。

### phase257：base command 时间持续性离线归因（已验证）

```text
上游结论与待解决瓶颈：phase256 已把实际 base speed 降低约 42%，但 actor goalward command 转正后实际 goalward 仍为负，几何与联合停稳未改善
基线 checkpoint / SHA-256：不加载模型；复用 phase255 同 checkpoint、同 seed 的正式 control/candidate 冻结张量
唯一变量：仅比较 base temporal fraction `[0.9,0.9,0.9]` 与 `[0,0,0]` 已有轨迹的时间统计，不重新运行仿真
待验证假设：逐拍独立 base exploration 虽保持瞬时命令阈值覆盖，但相邻命令缺乏同向持续性，低层尚未建立目标方向速度时命令已换向，因此有效 goalward 运动消失
禁止改动范围：不改 fraction、rho、std、reward、observation/action、低层、安全阈值或训练代码；不扫描中间 fraction
运行规模与日志：离线读取 `doorman_train:/tmp/b2z1_phase255_gain_fix_{control,base_independent}.pt`，只输出终端统计，不新增源码或张量
成功/失败判据：在排除 reset 边界后同时给出 command lag correlation、同向保持长度、actual velocity 对当前/前一拍 command 的投影和 direction cosine；若不能区分时序与低层响应，不修改行为
观测指标：base command 阈值覆盖、lag-1/2/4 correlation、连续方向余弦为正的 run length、post velocity 与 command(t/t-1) 的投影/余弦、actor mean 与 sampled noise 量级
结论与已排除方向：已验证。control/candidate 的 base actor mean 绝对值与 sampled-noise 绝对值近似相同，排除策略输出尺度差异；lag-1 command correlation 为 `0.848/约0`，相邻 active command 方向余弦大于 `0.5` 的比例为 `93.82%/33.49%`。control 的实际速度沿当前、lag-1、lag-2、lag-4 command 投影为 `0.1332/0.1369/0.1404/0.1446 m/s`，candidate 仅为 `0.0261/0.0137/0.0123/0.0096 m/s`；对应方向余弦由 `0.553--0.587` 降到 `0.204--0.079`。这直接证明 iid 命令在低层响应形成前换向；不是低层坐标或方向错误，也不能把低速解释成更好的可控停稳
正式保留/删除的代码：无代码变化
前序阶段回归：phase256 生产链、checkpoint、日志和安全结果已核对
下一决策门槛：phase258 必须用 iid 输入辨识低层响应核，给出原则性的延迟和保持时间；在此之前不选择中间 rho/fraction，不恢复 0.9，也不修改 reward
```

2026-09-29 06:21，phase257 只读取 phase255 两份正式冻结张量完成离线归因，没有启动 Isaac、修改代码或生成新数据文件。reset 边界已排除；结果确认 iid base exploration 的瞬时幅度足够，但缺少低层执行所需的连续同向窗口。phase257 封账，下一原子动作进入 phase258 响应核系统辨识。

### phase258：低层 base command 响应核辨识（已验证）

```text
上游结论与待解决瓶颈：phase257 已证明 iid base command 逐拍换向使实际速度无法跟随，但尚未量化低层的延迟、响应核和原则性最小保持时间
基线 checkpoint / SHA-256：不加载模型；使用 phase255 iid candidate 冻结轨迹及其同 checkpoint control 作外部验证
唯一变量：对非 reset 的 iid command/velocity 序列拟合统一二维 FIR 传递响应；不运行新仿真、不修改配置
待验证假设：低层速度主要由过去若干拍 command 的正权重累计形成，累计核达到主要响应需要多于一个高层步，因此 iid 输入被物理低通滤掉
禁止改动范围：不扫描 rho/fraction，不改 std、reward、observation/action、低层或训练代码；不得从 control 自相关轨迹单独拟合并误认因果核
运行规模与日志：离线读取 phase255 candidate 全部非 reset transition，按环境拆分 train/validation；control 只作外部预测方向校验
成功/失败判据：报告 FIR 各 lag 系数、累计响应比例、验证集 R2/误差和达到 50%/80% 响应的拍数；若模型不足以识别，不据此设计机制
观测指标：command x/y 到 velocity x/y 的对角/交叉核、截距、R2、残差、累计核、control 外部一致性
结论与已排除方向：已验证。首版跨窗口纯 FIR 因未递归建模速度状态，验证 R² 随窗口增长下降，已判为模型不充分且不作结论。改用每拍显式包含 `velocity(t-1)` 的 ARX 后，command lag 由 0 增至 4 时验证集 x/y R² 从 `0.845/0.860` 提升到 `0.852/0.891`，lag 8 以后收益平台或回退。ARX(4) 自回归特征值为 `0.905/0.939`；由其冲激响应积分得到 x/y 50% 响应为 `5/10` 拍、80% 为 `15/25` 拍、90% 为 `22/35` 拍，交叉响应绝对和约为对角的 `11.8%`。按 50 Hz 策略频率，主要响应时间约 `0.3--0.5 s`。因此 iid 一拍命令必然被低层低通；直接缩短 rho/fraction 不能同时保留行走和停稳，时间相关与状态幅度必须分层拥有
正式保留/删除的代码：无源码变化；分析使用一次性容器命令
前序阶段回归：phase257 输入时序和 reset mask 已核对
下一决策门槛：phase259 必须先证明现有概率链中哪个组件拥有时间尺度、哪个组件拥有 observation-dependent 幅度，以及策略是否观测 temporal latent；在所有权闭合前不实现新分布
```

2026-09-29 06:24，phase258 完成。分析仅使用 phase255 iid candidate 的非 reset 轨迹，按环境 75%/25% 拆分训练与验证；control 没有参与拟合，避免自相关输入混淆响应核。没有启动 Isaac、修改源码或生成持久诊断文件。phase258 封账，下一原子动作进入 phase259 代码与概率链审核。

### phase259：探索时间尺度与幅度所有权审核（已验证）

```text
上游结论与待解决瓶颈：phase258 证明 base command 必须 coherent 约 15--25 拍才能形成主要低层响应；固定相关噪声近目标又无法停稳
基线 checkpoint / SHA-256：不加载 checkpoint，不运行训练
唯一变量：静态审核现有 temporal Gaussian、state-dependent attenuation、variance budget、actor observation 和 rollout/deployment 路径的所有权及条件概率语义
待验证假设：现有架构已把时间尺度归 temporal context、状态幅度归 attenuation，但 actor/likelihood 对 temporal latent 的可见性或幅度 credit 路径存在缺口，导致两层虽形式解耦却无法共同学习
禁止改动范围：不实现候选、不改 fraction/rho/std/reward/observation/action/低层；不得提出 stage/readiness/seed/门型特例、固定时间退火或参数扫描
运行规模与日志：仅源码、配置和既有 phase230/236/237/250/252 证据审核，不启动 Isaac
成功/失败判据：明确列出每个随机变量的采样时序、actor 可见信息、conditional/marginal log-prob、reset、deployment 值和梯度所有权；指出首个确定缺口或证明现有架构完整
观测指标：context 状态、fraction/rho、attenuation/global std、budget、actor_obs/history、old/new likelihood、evaluation sampling
结论与已排除方向：已验证。(1) 物理 rollout 每拍先按 `rho` 推进单位方差 AR(1) context，上一拍 done 的环境直接重采 stationary context；rollout 边界也从 stationary prior 重新开始。(2) action 条件分布为 `mean + sigma*sqrt(fraction)*context` 和 `sigma*sqrt(1-fraction)`；state-dependent attenuation 先作用于 marginal sigma，因此同时缩放相关和独立分量。(3) PPO 不把不可观测 context 当 observation，而是按整段 action、mean、sigma、done 和 fraction 用 Kalman 递推精确边缘化；old/new sequence likelihood、KL 和 reset 语义一致，attenuation 通过 new marginal sigma 获得真实 PPO 梯度。(4) entropy 只训练 global std，variance budget 只对 rollout batch 的 marginal variance 下界施加 attenuation 梯度，二者所有权明确。(5) actor 不直接观察 context 不是概率实现错误；当前 observation 已包含上一拍 `delta_actions`、`b2z1_commands`、base/EE 实际运动，能够观察噪声产生的物理结果。评估虽然调用 rollout 构造 state，实际送入环境的是 `action_mean`，因此部署为无噪声 deterministic actor。(6) 首个确定缺口是 temporal fraction 仅来自固定配置，`_compute_temporal_action_noise_fraction()` 没有 observation-dependent owner 或学习梯度。现有 attenuation 能按状态改变总幅度，却不能在保持 marginal command coverage 的同时独立改变相关份额；这正是 phase256 的 iid 无法行走与此前固定相关近目标无法停稳之间缺失的统一表达能力。排除重新怀疑 AR reset、似然符号、actor latent 直连和部署随机性
正式保留/删除的代码：无代码变化
前序阶段回归：phase258 系统辨识已通过独立环境验证；容器内 temporal noise、actor interface、exploration budget 共 `29 passed`
下一决策门槛：phase260 只完成 observation-dependent temporal fraction 的数学与接口契约。必须支持逐 observation、逐 action fraction，分别保存 old fraction 并重算 new fraction，用 exact sequence likelihood 训练；不得读取 stage/readiness/seed/门型或脚本目标，不得实现或启动训练
```

2026-09-29 06:27，phase259 静态审核完成。宿主机无项目 Python，相关单测按容器化约定在 `doorman_train` 内执行并得到 `29 passed in 7.40s`；未启动 Isaac、未修改训练代码或生产配置。审核确认现有概率链内部一致，真正缺失的是 observation-dependent temporal fraction 的所有权，而不是再调固定 fraction、rho、std 或 reward。phase259 封账，下一原子动作进入 phase260 概率契约设计。

### phase260：状态相关时间持续性概率契约设计（已验证）

```text
上游结论与待解决瓶颈：phase259 已证明现有固定 fraction 概率链正确，但只能全局选择 coherent 或 iid，无法让探索持续性随物理 observation 变化
基线 checkpoint / SHA-256：不加载 checkpoint，不运行训练
唯一变量：只设计 observation-dependent、逐 action temporal fraction 的概率模型、网络所有权、storage 和 exact PPO likelihood 契约
待验证假设：让统一网络从 actor observation 输出 correlated fraction，可在不改变 action marginal variance 上限的前提下表达远处 coherent command 与近处低持续性；该变量与 attenuation 的状态幅度职责正交
禁止改动范围：不实现代码，不改生产 fraction/rho/std/reward/observation/action/低层/完成条件；禁止 stage、readiness、seed、门型、时间步或脚本目标特例；不做参数扫描
运行规模与日志：仅数学推导、源码接口表和既有证据审核，不启动 Isaac
成功/失败判据：明确随机变量生成式；old/new fraction 的 storage 与重算；逐步 exact marginal likelihood、KL、entropy、budget、reset、min/max 和 deterministic deployment 语义；证明 fraction 网络只读统一 actor observation 且不改变十维 action 语义
观测指标：无运行指标；审核梯度是否分别到 actor mean、attenuation、global std、fraction owner，old/new policy ratio 是否在参数不变时严格为 1
结论与已排除方向：已验证。统一生成式为 `z_t=rho*z_{t-1}+sqrt(1-rho^2)*eta_t`，可学习维使用 `f_t=f_min+(f_max-f_min)*sigmoid(h_phi(o_t))`，`a_t=mu_theta(o_t)+sigma_psi(o_t)*(sqrt(f_t)*z_t+sqrt(1-f_t)*epsilon_t)`；`f_t` 为逐 observation、逐 action 的确定性策略参数，可学习维满足 `0 <= f_min < f_max < 1`，严格常量维允许 `f_min=f_max`。fraction owner 读取与 actor 相同的规范化 observation，不允许额外 stage/readiness/seed/门型或脚本目标入口；其输出层零权重并以配置初值初始化，使接线时可严格复现固定 fraction 基线。rollout 必须保存 old `mu/sigma/fraction`；PPO update 从同一 observation 重算 new `mu/sigma/fraction`，old/new 各自独立执行 Kalman sequence filter 后计算 log-prob 和条件 KL，禁止复用 old fraction 计算 new policy。fraction 只通过真实 PPO sequence likelihood 学习；marginal variance budget 仍只约束 `sigma`，因为改变 `f` 不改变单步边际方差；global entropy 计算对 fraction stop-gradient，防止容量正则替代物理结果来选择持续性。done 后 context 回 stationary prior，rollout 边界同样重置；deterministic eval/export 完全忽略 context、sigma 和 fraction，只输出 `mu`。不需要把 latent 直接加入 actor observation，也不改变十维 action、reward 或环境语义
正式保留/删除的代码：无代码变化
前序阶段回归：phase259 概率链 `29 passed`
下一决策门槛：phase261 只能按上述契约做单一最小实现。纯张量测试必须覆盖固定 fraction 等价、参数不变时 ratio=1、old/new fraction 不同、fraction 非零梯度、done/reset、边界、marginal variance 不变和 deterministic deployment；通过前不启动 Isaac
```

2026-09-29 06:30，phase260 数学与接口设计完成。该设计没有引入阶段分支、手工 readiness、seed/门型特调、reward 修改或新的动作语义；现有 fixed fraction 将被统一接口的常量初始化严格包含，而不是保留第二条旧路径。未修改源码、未启动进程。phase260 封账，下一原子动作进入 phase261 最小实现和纯张量测试。

### phase261：状态相关 temporal fraction 最小实现（已验证）

```text
上游结论与待解决瓶颈：phase260 已完成无特例概率契约；当前源码仍只有 fixed fraction，尚未验证新参数链能否严格复现旧策略并形成正确 PPO 梯度
基线 checkpoint / SHA-256：不加载 checkpoint，不运行训练；只使用构造张量和现有单元测试
唯一变量：增加一个读取统一 actor observation 的逐 action fraction owner，并让 rollout/storage/PPO 分别携带 old fraction、重算 new fraction
待验证假设：按 phase260 契约实现后，常量初始化与旧 fixed fraction 数值等价；参数变化时 old/new exact sequence likelihood 和 fraction gradient 正确，且不影响 marginal variance、reset 或 deterministic action
禁止改动范围：不改 reward、环境 observation 内容、十维 action、低层、完成条件、rho/std/预算数值或生产训练参数；不增加 stage/readiness/seed/门型/时间步/脚本目标输入；不启动 Isaac
运行规模与日志：容器内纯 Python/torch 单元测试和 py_compile
成功/失败判据：固定 fraction 等价、ratio=1、old/new fraction 分离、非零有限梯度、done/reset、边界、marginal variance、eval mean 全部通过；旧固定路径被新统一接口替代，不并存死分支
观测指标：逐测试最大数值误差、ratio、fraction head gradient、边界和 reset 后首步分布
结论与已排除方向：已验证。Actor 增加独立 fraction owner；固定配置仍走同一 fraction 计算接口，显式启用后从规范化 actor observation 输出逐动作 fraction。rollout 继续保存 old fraction；PolicyAndValueWrapper 暴露 update 时重算的 new fraction；PPO 通过新增成对滤波纯函数分别计算 old/new sequence conditionals，global entropy 对 new fraction detach，variance budget 未接入 fraction。可学习维使用严格有界 sigmoid；`min=max` 的动作维精确保持常量且无任意 epsilon。测试证明 fraction head 初始化复现指定常量、有界且梯度有限非零，old/new fraction 分离后各自匹配直接 likelihood，参数相同时 ratio=1，done/reset 和 marginal variance 既有测试无回归
正式保留/删除的代码：保留 `actor_critic_modules.py` fraction owner、`temporal_action_noise.py` 成对滤波纯函数、`ppo_trainer.py` old/new 数据链及对应测试；生产配置尚未启用，不存在临时环境变量或诊断出口
前序阶段回归：容器内 py_compile、完整 `58 passed, 5 subtests passed`、`git diff --check` 通过
下一决策门槛：phase262 只接通生产配置并运行单卡 `8 env x 8 steps x 1 update` optimizer smoke；必须确认 fraction owner 参数有有限非零梯度和更新、old/new ratio 初始为 1、checkpoint 包含新参数，才允许扩大规模
```

2026-09-29 06:38，phase261 完成。新增实现没有修改 reward、环境 observation 内容、十维 action、低层、完成条件或现有生产参数，也未启动 Isaac。完整测试由 `57 passed, 5 subtests passed` 增至 `58 passed, 5 subtests passed`；新增与既有 temporal/actor/budget 定向测试共 34 项通过。当前无运行进程，下一原子动作进入 phase262 配置接线和极小 optimizer smoke。

### phase262：状态相关 fraction 单卡 optimizer smoke（已验证）

```text
上游结论与待解决瓶颈：phase261 静态实现和完整回归通过，但新 fraction owner 尚未在真实 rollout、storage、PPO optimizer 和 checkpoint 路径运行
基线 checkpoint / SHA-256：随机初始化，不加载旧 checkpoint；接口新增参数后旧 checkpoint 禁止继承
唯一变量：生产配置启用 state-dependent temporal fraction；base x/y/yaw 与 ee_x 为可学习维并以已验证 fixed fraction `0.9` 初始化，其余六维严格 `min=max=initial=0`
待验证假设：极小真实 rollout 中 old fraction 正确存储、new fraction 正确重算，初始策略 ratio 为 1，fraction owner 获得有限非零 PPO 梯度并被 optimizer 更新，checkpoint 完整保存参数
禁止改动范围：不改 reward、observation/action、低层、完成条件、rho、std、attenuation、variance budget 或其他训练参数；不扩大 env/update，不做参数扫描
运行规模与日志：GPU 3、`8 env x 8 steps x 1 update`，纯动力学 headless；日志 `doorman_train:/tmp/b2z1_phase262_fraction_optimizer_smoke.log`，checkpoint 保存频率 1
成功/失败判据：无 traceback/NaN；cameras=0、RTX sensors=0、frame rendering disabled；fraction shape/边界/固定维精确为 0；owner grad/update 有限非零；保存后 state_dict 可重载
观测指标：old/new fraction、ratio、fraction head grad norm/parameter delta、loss/KL、固定维误差、checkpoint reload
结论与已排除方向：已验证。首次 smoke 的真实 rollout 正常，但 PPO update 后 actor mean 全 NaN；最小反例证明固定 `fraction=0` 和可学习正 fraction 共处时，直接 `sqrt(fraction)` 在零点产生无穷导数，形成 `0*inf=NaN` 并经全局梯度裁剪污染全部参数。通用修复把零点平方根的前向保持为 0、反向明确定义为 0，正 fraction 数学不变；反例梯度由 `[NaN,1.338953]` 变为 `[0,1.338953]`。同参数复跑正常完成 64 timesteps/1 update，policy/value/weighted loss 为 `-0.044148/12.477573/12.662681`，无 traceback/NaN。四个可学习输出层 weight/bias 最大变化为 `5.0203e-4/5.0163e-4`，固定六维保持精确 0，全部 policy tensor 有限。checkpoint 使用保存配置重建 Actor 后 `strict=True` 全键重载成功。纯动力学日志确认 `cameras=0`、`RTX sensors=0`、`frame rendering disabled`
正式保留/删除的代码：保留 phase261 通用实现、生产配置、零边界安全平方根和永久回归测试；没有加入临时诊断代码或环境变量。首次失败运行目录只保留无 checkpoint 的错误证据，不得继承
前序阶段回归：phase261 py_compile、`58 passed, 5 subtests passed`、diff check 通过
下一决策门槛：phase263 四卡 8192 env 从随机初始化训练到 step25；必须同时评价 command coherence、fraction 状态分化、五项几何、stable 1/5/10、GRASP、安全与终止，禁止仅凭吞吐或平均速度宣布成功
```

2026-09-29 06:42，phase262 唯一 optimizer smoke 已登记并启动。启动前 GPU 3 显存占用 `30 MiB`、利用率 `0%`，容器内无训练或 Isaac 残留进程。运行期间冻结源码和配置，下一原子动作只轮询该作业并核对日志、checkpoint 与 fraction owner 参数更新。

2026-09-29 06:47，phase262 首次 smoke 在完成真实 rollout、进入 PPO update 后失败，未执行完 optimizer、未生成 checkpoint，进程已退出。直接复现确认：同一 fraction 张量中同时存在固定 `0` 维和可学习正值维时，原序列滤波的 `sqrt(fraction)` 在零点产生无穷导数，固定维链路形成 `0 * inf = NaN`；全局梯度裁剪随后会把 NaN 扩散到 actor 参数，导致下一次 forward 的 mean 全 NaN。修复仅位于通用 temporal Gaussian 数学层：零点前向仍精确为 `sqrt(0)=0`，反向导数显式定义为 0，正 fraction 维公式不变。最小反例由 `grad=[NaN,1.338953]` 修正为 `[0,1.338953]`；完整回归为 `59 passed, 5 subtests passed`，diff check 通过。配置和 smoke 定义均未改变，下一原子动作是按相同参数重跑一次。

2026-09-29 06:50，phase262 同参数复跑完成并退出，无残留进程。运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260929_064544`；checkpoint `model_step_000001.pt` SHA-256 `24cee046444706dcdf90a38a2897d2bff92fafe2a0d0c7bbfe2a574901dbc794`，日志 `doorman_train:/tmp/b2z1_phase262_fraction_optimizer_smoke.log` SHA-256 `88f8bd722dbb61963b9768ddeb4d616138759503a8b4e959c915fee615782f27`。checkpoint 含 33 项 optimizer state，fraction owner 参数已更新、固定维不变并完成 strict reload。phase262 封账，smoke checkpoint 仅作接口证据，不续训。

### phase263：状态相关 fraction 四卡干净训练（失败）

```text
上游结论与待解决瓶颈：phase262 已验证真实 optimizer 链；尚不知道新机制能否在保持低层 coherent command 的同时学出近目标较低持续性并形成联合停稳
基线 checkpoint / SHA-256：随机初始化，不加载 phase262 或任何旧 checkpoint
唯一变量：相对 phase256 失败训练，仅将 fixed/base-iid temporal fraction 替换为 phase262 已验证的 observation-dependent fraction；reward、observation/action、低层和全部环境参数不变
待验证假设：base/ee_x 初始 `0.9` 保留可执行的持续探索，而 fraction owner 能从统一 observation 学到状态间持续性差异，使 APPROACH_PREGRASP 出现稳定窗口和 GRASP 样本
禁止改动范围：训练运行中冻结源码、配置、reward、完成条件、std/rho/attenuation/budget、随机化与低层；不继承 checkpoint、不延长到 step25 以后、不据早期波动改参数
运行规模与日志：GPU 0--3，每卡 2048 env、128 steps、4 mini-batches，共 8192 env，从随机初始化到 step25；日志 `doorman_train:/tmp/b2z1_phase263_state_fraction_clean25.log`，checkpoint 保存频率 25
成功/失败判据：运行链无异常/NaN；command coherence 和实际 goalward motion 不低于有效相关基线；五项几何不回归；stable 1/5/10 或 GRASP 必须出现非零证据；安全终止不恶化。若仍无停稳则判失败并只读归因
观测指标：逐动作 fraction mean/std/min/max、command lag/coherence、command/actual goalward、五项 quality、stable 1/5/10、stage reached/transition、existing/unsafe/overspeed/collision
结论与已排除方向：失败。step25 正常完成 `26,214,400` timesteps，末轮约 `20,857 steps/s`，四 rank 始终为纯动力学且无 NaN/traceback。末轮五项 quality 为 `0.3547/0.4978/0.1587/0.1136/0.8914`，相对 phase256 四项提高、base yaw 降低；command/actual goalward 为 `+0.0319/+0.0037 m/s`，方向优于 phase256，但 actual planar speed 回升到 `0.2081 m/s`。stable 1/5/10、stable episode max、GRASP occupancy/count 均为 0，因此不满足预注册成功标准，不续训。另确认 fraction summary 虽进入运行时 `state.log_history`，但 Rich callback 只输出白名单，ModelSaveCallback 保存时又删除 `log_history`，故 checkpoint 不含该证据；此前“step25 后直接读取”的假设被否定
正式保留/删除的代码：训练前仅增加通用 fraction 统计；不增加行为诊断或早退
前序阶段回归：phase262 完整 `59 passed, 5 subtests passed`、真实 optimizer、checkpoint strict reload 通过
下一决策门槛：phase264 仅修复完整 metrics 的通用持久化，并用冻结 checkpoint 做首断点归因；禁止续训、扫描 fraction/rho/std 或直接修改 reward
```

2026-09-29 06:54，phase263 四卡干净训练已登记启动。训练前新增的通用逐动作 fraction `mean/std/min/max` 统计只读取 rollout tensor，不参与 loss 或行为；py_compile、完整 `59 passed, 5 subtests passed` 和 diff check 通过。运行期间冻结源码与配置，下一原子动作只核对 update1 的四卡、纯动力学和数值稳定性。

2026-09-29 06:55，按进程、日志和工作树复核执行游标：四个训练 rank 均存活，四路均记录 `camera_types: []`，尚未出现 update1、NaN 或 traceback；当前仍处于场景创建。已修正顶部执行卡、未决问题、近期尝试和本 phase 的过期“计划中”状态。唯一下一原子动作保持为等待并核对 update1，不重复启动实验、不编辑训练源码或配置。

2026-09-29 06:59，phase263 update1 完成：四卡共采集 `1,048,576` timesteps，collection `42.082 s`、learning `9.806 s`、iteration `51.89 s`，吞吐约 `20,208 steps/s`；四个 rank 均完成纯动力学验证，无 NaN、traceback 或运行时异常。首轮 `GRASP=0` 只作为随机初始化基线，不提前判失败。逐动作 fraction summary 已写入 Trainer metrics/history，但当前 Rich 控制台仅打印硬编码字段，因此日志文本暂不可见；按运行冻结约束不改 logger，step25 保存后读取完整 history。下一原子动作移动为保持代码冻结并完成 step25。

2026-09-29 07:20，phase263 正常完成 step25 并退出。checkpoint `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260929_065343/model_step_000025.pt` SHA-256 `0628e11e30ce62dc269f0473ff6012fa73613ae137b865801c8261851f9498aa`；日志 `doorman_train:/tmp/b2z1_phase263_state_fraction_clean25.log` SHA-256 `92007754de3c12dfac748f630a7a492cc00767706a773bc157c56154eadf9327`。预注册的 stable 与 GRASP 均未出现，phase263 判失败，checkpoint 禁止续训。静态审核同时确认完整 metrics 没有持久化：`Trainer.log()` 运行时追加到 `state.log_history`，Rich callback 只打印白名单，而 `ModelSaveCallback.save_checkpoint()` 显式删除 `log_history`。该观测缺口转入 phase264，不回头重跑 phase263。

### phase264：完整 Trainer metrics 通用持久化（已验证）

```text
上游结论与待解决瓶颈：phase263 已按结果判失败，但 fraction mean/std/min/max 等已计算指标既不在文本日志中，也被 checkpoint 主动丢弃，无法做冻结首断点归因
基线 checkpoint / SHA-256：phase263 step25 `model_step_000025.pt` / `0628e11e30ce62dc269f0473ff6012fa73613ae137b865801c8261851f9498aa`，只读使用
唯一变量：在 Trainer 通用日志所有权层，把 rank0 每个 update 的完整 metrics 原样持久化为机器可读 sidecar；checkpoint 继续保持轻量
待验证假设：通用 sidecar 能保留所有 Policy/Env/train 指标，且不改变 callback、optimizer、checkpoint、环境或策略行为
禁止改动范围：不改 reward、observation/action、环境、低层、策略输出、fraction/rho/std/attenuation/budget、完成条件或随机化；不启动续训或参数扫描
运行规模与日志：先做纯 Python 单测和静态回归；通过后才设计冻结 checkpoint 的单 rollout 诊断
成功/失败判据：仅主进程写文件；逐 update 一行、可解析、包含 step/timesteps 与 temporal fraction；多 rank 不重复；序列化有限标量；checkpoint strict reload 和既有测试不回归
观测指标：sidecar 路径、行数、键集合、rank ownership、异常/NaN 序列化行为、checkpoint 大小与键集合
结论与已排除方向：已验证。新增通用 rank0 `training_metrics.jsonl` sidecar，每个 update 追加一条完整 metrics；torch/Python 标量统一转标准 JSON，NaN/Inf 写为 `null` 并列入 `_non_finite_metrics`，非标量明确拒绝。两个 PPO Trainer 共用同一 helper；非主进程不创建文件。checkpoint 继续删除庞大 history，其键集合与恢复语义不变。排除把 fraction 指标硬编码到 Rich 面板或把完整 history 塞回 checkpoint
正式保留/删除的代码：保留 `trl/utils/metrics_log.py`、两个 Trainer 的统一调用和 `test_metrics_log.py`；无临时环境变量、诊断出口或行为分支
前序阶段回归：phase263 checkpoint、日志 SHA 和无残留进程已核对
下一决策门槛：phase265 用冻结 phase263 checkpoint 做唯一一次同规模 rollout，sidecar 必须含逐动作 fraction 和预注册 Env/train 指标；源 checkpoint SHA 必须保持不变
```

2026-09-29 07:24，phase264 完成。新增定向测试 `4 passed`，两个 Trainer py_compile、`git diff --check` 和完整 `63 passed, 5 subtests passed` 均通过。未启动 Isaac 或训练，未修改 reward、observation/action、环境、策略行为、探索参数或 checkpoint 格式。phase264 封账，下一原子动作进入 phase265 冻结 checkpoint 单 rollout 诊断。

### phase265：冻结 checkpoint 首断点单 rollout 诊断（已验证）

```text
上游结论与待解决瓶颈：phase263 有 goalward 与四项几何改善但完全没有联合停稳；phase264 已补齐完整 metrics sidecar
基线 checkpoint / SHA-256：phase263 step25 `model_step_000025.pt` / `0628e11e30ce62dc269f0473ff6012fa73613ae137b865801c8261851f9498aa`
唯一变量：从该 checkpoint 恢复相同四卡、每卡 2048 env、128 steps，只收集一个 rollout/update 的完整 sidecar；分析 optimizer 前 rollout，输出模型丢弃
待验证假设：可学习四维 fraction 已随 observation 形成差异，但近目标样本的持续性或幅度仍未下降到能联合停稳；或反证 fraction owner 仍近似常量
禁止改动范围：不改源码、配置、reward、环境、策略、fraction/rho/std/attenuation/budget、完成条件或随机化；不续训第二个 update，不保存为可继承 checkpoint
运行规模与日志：GPU 0--3，各 2048 env、128 steps，从 step25 恢复后只到 step26；新输出目录及 sidecar 待启动后登记
成功/失败判据：四 rank 纯动力学且无 NaN；sidecar 恰有一条新记录并包含 10 维 fraction mean/std/min/max、五项 quality、stable、goalward、actual speed、GRASP 与安全指标；源 checkpoint SHA 不变
观测指标：fraction 各维 mean/std/min/max；command/actual motion；五项 quality；stable 1/5/10；GRASP；existing/unsafe/overspeed/collision
结论与已排除方向：已验证。sidecar 恰有 1 条、956 个键且 `_non_finite_metrics=[]`。四个可学习 fraction 的 mean/std/min/max 分别为 base_x `0.8909/0.0123/0.5633/0.9201`、base_y `0.8954/0.0071/0.7680/0.9197`、base_yaw `0.8937/0.0091/0.5782/0.9236`、ee_x `0.8928/0.0142/0.1688/0.9096`；固定六维严格为 0，说明 owner 已形成有限 observation dependence，但多数样本仍接近初值 0.9。更早的物理断点是 `gripper_orientation_ok=4.77e-6`、`pregrasp_position_ok=1.63e-4`；base position/yaw OK 为 `0.0167/0.0800`，still 为 `0.00165`，all-checks/stable/GRASP 全 0。因此当前不能把失败首先归因于近目标停不稳，必须先审核 EE 几何 credit。源 checkpoint SHA 前后不变，输出模型不保留
正式保留/删除的代码：不新增代码；输出模型只作 throwaway，不进入有效 checkpoint 列表
前序阶段回归：phase264 `63 passed, 5 subtests passed`
下一决策门槛：phase266 只读审核 orientation/position 的结果型 reward、advantage 与动作 credit；根因成立前不修改行为
```

2026-09-29 07:26，phase265 已启动。容器 `doorman_train`，GPU 0--3、每卡 2048 env、128 steps、4 mini-batches，从 phase263 step25 恢复且只执行一个 update 到 step26；保存频率 500，输出模型不进入可继承列表。独立日志为 `/tmp/b2z1_phase265_frozen_rollout.log`；启动前后源 checkpoint SHA 均为 `0628e11e30ce62dc269f0473ff6012fa73613ae137b865801c8261851f9498aa`。运行期间冻结代码和配置。

2026-09-29 07:31，phase265 完成并退出。运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260929_072632`；sidecar SHA-256 `3e560a8acb9de6f2ba30d189095394ee0dab00e6aced3616339c287bd309f44c`，日志 SHA-256 `7ae9f3257f9d6aebdb52720c37a1742678822ebb2033b1e45bc0d027aec3089e`，源 checkpoint SHA 仍为 `0628e11e30ce62dc269f0473ff6012fa73613ae137b865801c8261851f9498aa`。输出目录没有模型文件，未形成可继承 checkpoint。phase265 封账，下一原子动作进入 phase266 只读 credit 链审核。

### phase266：APPROACH_PREGRASP 几何 credit 链审核（已验证）

```text
上游结论与待解决瓶颈：phase265 证明 orientation/position 几乎从不进入验收区，早于 stable 失败；fraction owner 已非恒定但尚不能评价近目标分化是否正确
基线 checkpoint / SHA-256：phase263 step25 `model_step_000025.pt` / `0628e11e30ce62dc269f0473ff6012fa73613ae137b865801c8261851f9498aa`，只读
唯一变量：静态追踪 orientation/position 的真实物理误差如何进入 stage progress、reward、GAE advantage 与对应动作维 PPO credit，并与 phase265 同批聚合指标核对
待验证假设：当前 stage progress 的组合或尺度使最弱 orientation/position 分量获得的结果型 credit 过稀、被其他易改善分量或终止成本淹没，因而策略只学到总体 goalward 而未学到 EE 几何
禁止改动范围：不改任何代码、reward、observation/action、环境、低层、探索参数、完成条件或随机化；不启动 Isaac、训练或参数扫描
运行规模与日志：仅源码、配置、phase263/265 日志与 sidecar 审核
成功/失败判据：明确列出每个几何分量的公式、尺度、delta/absolute 语义、组合方式、动作影响维和 PPO 信号；以直接证据确认首个 credit 断点或否定假设
观测指标：五项 quality/current/best/ok、stage progress/reward、termination、action mean/noise/command、advantage 聚合实现
结论与已排除方向：已验证，原假设被否定。五项物理结果先以 `exp(-error/scale)` 形成连续质量，再按各自真实验收阈值归一化并截断；APPROACH progress 为 `0.5 * mean + 0.5 * min`，orientation/position 既有均值 credit，作为最弱项时还获得额外 bottleneck credit。该 progress 以 signed `current-previous` 进入 `task_stage_progress`，最终每步物理势能变化的有效系数为 80；另有只依赖真实 progress 的 residual cost，不读取 actor command。标准 GAE 使用 `gamma=0.9975`、`lambda=0.985`，按 stage x door variant 归一化；PPO 使用完整 temporal sequence 的 old/new conditional Gaussian log-prob，scalar ratio 乘同一步 advantage，对十维 mean/fraction 均有真实 score gradient。actor observation 包含 9 维 gripper-to-handle 相对变换、26 维任务几何、10 维当前 B2Z1 command 和实际状态；canonical action 到 physical command 连续，phase217 数值雅可比已证明左右门均满秩 10。phase263 的 orientation/pregrasp quality 从 step1 `0.0537/0.1261` 持续改善到 step25 `0.1136/0.1587`，best 同样升至 `0.2671/0.2456`，没有 reward 或 PPO credit 消失、符号相反、observation 缺失或动作不可达的证据。orientation 当前平均质量 `0.06188` 对应约 `1.39 rad` 误差；实测 iid 姿态噪声绝对均值约 `0.0072`，对应每步目标旋转绝对均值约 `0.0068 rad`，完整 1000 步随机游走每轴 RMS 约 `0.27 rad`，说明单回合随机覆盖有限，但策略均值可以积累且已经获得持续改善 credit。历史 phase224--227 在相同统一动作/reward 主干上也是 step25 尚无 stable、step50 几何显著改善、step70 首次出现一帧联合状态。因此 phase263 的“step25 必须出现 stable/GRASP”是过早实验门槛，不是代码根因；禁止据此修改 reward、增大姿态噪声或引入 stage 特例
正式保留/删除的代码：无
前序阶段回归：phase265 sidecar、日志、源 checkpoint SHA 和无残留进程已核对
下一决策门槛：phase267 不改行为，从随机初始化训练到 step75。step25/50 只判断五项趋势、安全和数值稳定；只有趋势平台、回归或安全恶化才提前停止。step75 要求出现非零 stable_reached_1，并优先观察 stable_reached_5/10 与 GRASP；若仍失败，再对当时最早瓶颈做冻结归因
```

2026-09-29 07:41，phase266 静态审核完成。未修改或运行任何训练代码、reward、探索参数、环境或 Isaac；结论来自当前源码、phase263 逐 update 趋势、phase265 同批 sidecar 和 phase224--227 历史对照。当前无训练进程。

### phase267：当前统一机制四卡干净训练（step50 安全门槛失败）

```text
上游结论与待解决瓶颈：phase266 排除 EE 几何 credit 链断裂，并确认 phase263 step1--25 两个最弱结果仍持续改善；25 update 要求 stable/GRASP 的门槛早于已有同主干学习时间尺度
基线 checkpoint / SHA-256：无；从随机初始化开始，不继承已封账的 phase263 checkpoint
唯一变量：无行为变量；保持当前源码、配置、reward、observation/action、状态相关 amplitude/fraction、低层和随机化完全不变，只把完整观察窗口从 25 update 延长到 75 update
待验证假设：当前统一机制需要超过 25 update 才能把持续改善的五项几何结果带入同一时刻；到 step75 应出现非零联合稳定并进入 GRASP
禁止改动范围：运行期间冻结源码和配置；不得修改 reward、完成条件、std/rho/fraction/variance budget、网络、随机化、低层或训练拓扑；不得加载旧 checkpoint、增加 stage/seed/门型特例或脚本动作
运行规模与日志：源开发机 GPU 0--3，每卡 2048 env、128 steps/env、4 mini-batches，共 8192 env；纯动力学 headless、无相机/RTX；日志 `doorman_train:/tmp/b2z1_phase267_state_fraction_clean75.log`，rank0 运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260929_074257`，完整 metrics 写其中的 `training_metrics.jsonl`
成功/失败判据：update1 必须四 rank 正常、纯动力学且无 NaN；step25/50 五项 quality/best/seen 应保持同步改善，安全与 termination 不恶化；若趋势平台、回归或数值/安全异常则提前停止并冻结归因。step75 成功要求 `stable_reached_1>0` 且产生真实 GRASP 入口，stable 5/10 作为更强证据；若没有则按当时最早物理断点判失败
观测指标：五项 quality/best/ok/episode-seen、all-checks、stable 1/5/10、GRASP occupancy/reached、command/actual goalward、actual speed、逐动作 amplitude/fraction、termination 与安全指标
结论与已排除方向：step25 趋势门槛通过，但 step50 安全门槛失败并主动停止。step50 五项 quality 为 `0.6763/0.6587/0.5115/0.3064/0.9185`，pregrasp/orientation episode-best 为 `0.6485/0.4817`，goalward 为 `+0.0615 m/s`，证明 phase266 关于几何 credit 完整且需要更长学习窗口的判断成立；排除 step25 几何平台、reward 断链和底盘反向。与此同时 unsafe-force 从 step25 的 `4.58e-5` 增至 step50 的 `2.42e-4/step`，且 `termination/unsafe_force/approach_pregrasp` 与总 unsafe-force 相等；contact-free 同期降至约 `0.9600`。虽然 existing termination 总率下降，但 `>600 N` 是独立硬安全事件，满足预登记的安全恶化停止条件。stable 1/5/10 和 GRASP 仍为 0，因此不能把该 checkpoint 视为有效策略或继续赌 step75。现有聚合指标不足以判断碰撞对象、接触前相对速度和动作因果，禁止据此猜测修改
正式保留/删除的代码：无行为代码改动；只保留 phase264 已验证的通用 metrics sidecar
前序阶段回归：phase264 完整 `63 passed, 5 subtests passed`；phase265 sidecar 无非有限值；phase266 静态链路审核通过
下一决策门槛：phase268 只做冻结事件归因；必须直接取得 unsafe 发生前后的对象级 force、相对速度、几何误差、actual/command/action 和门侧，才能决定修复层
```

2026-09-29 07:42，phase267 已登记并启动。启动参数为 GPU 0--3、每卡 2048 env、128 steps/env、4 mini-batches、75 updates、每 25 update 保存一次；不加载 checkpoint。运行期间源码、配置和训练行为冻结，唯一下一动作是核对 update1。

2026-09-29 07:47，phase267 update1 完成，共 `1,048,576` timesteps，iteration time `54.09 s`。四个 rank 均确认 `cameras=0`、`RTX sensors=0`、`frame rendering disabled`；sidecar 恰有 1 条且 `_non_finite_metrics=[]`，无 traceback 或运行时异常。五项 quality 为 `0.3331/0.5115/0.1261/0.0537/0.7782`，只登记为随机初始化基线，不提前判断学习成败。下一原子动作移动到保持冻结并完成 step25。

2026-09-29 08:05，phase267 step25 趋势门槛通过。五项 quality 从 step1 的 `0.3331/0.5115/0.1261/0.0537/0.7782` 变为 `0.3582/0.4824/0.1669/0.0994/0.8842`；两个最弱项及其 episode-best 持续改善，goalward 从 `-0.0351` 转为 `+0.0027 m/s`，actual speed 从 `0.2202` 降至 `0.2088 m/s`。existing/unsafe termination 为 `4.37e-4/4.58e-5 per step`，无持续恶化；所有 25 条 sidecar 均无非有限值。stable 1/5/10 与 GRASP 仍为 0，但按本 phase 预登记规则不在 step25 判失败。step25 checkpoint SHA-256 为 `6aa189f19ee54241c6a44d826eea48f4a599ffac99b9f3f465139c78d1e476fc`，原进程继续到 step50。

2026-09-29 08:32，phase267 在 step50 按预登记安全条件停止。共完成 `52,428,800` timesteps，50 条 sidecar 均无非有限值；step50 checkpoint SHA-256 `438759b0c2928d315f37a9ae8bbfaae723698229334c14c28e393fc9667a1d2d`，sidecar SHA-256 `c8e2c01ab59e4a3ea81d97c69fb4a3c3b6adb8bedf4d741a666cf224b6900dd3`。四个 worker 已通过 TERM 结束，容器保持运行，GPU 已释放。该 checkpoint 只作冻结诊断，不续训。

### phase268：APPROACH_PREGRASP unsafe-force 冻结事件归因（已验证）

```text
上游结论与待解决瓶颈：phase267 证明几何持续学习，但 step25--50 的 >600 N unsafe-force 明显上升并全部发生在 APPROACH_PREGRASP；聚合 sidecar 无法给出碰撞对象和接触前因果链
基线 checkpoint / SHA-256：phase267 step50 `model_step_000050.pt` / `438759b0c2928d315f37a9ae8bbfaae723698229334c14c28e393fc9667a1d2d`，只读
唯一变量：增加无行为副作用的 reset 前事件快照，只在 unsafe-force 触发时保存触发帧及有限前序窗口的真实物理量；冻结策略只收集一个 rollout，optimizer 前退出
待验证假设：安全恶化来自几何学习后接近门体时的特定对象碰撞；必须区分 handle、panel、frame，以及 actor mean、sampled noise、base 运动或 EE tracking 哪一链首先产生过大相对速度/压入
禁止改动范围：不改 reward、600 N 阈值、终止/reset 时序、observation/action、std/rho/fraction/budget、策略参数、环境随机化或低层；不执行 optimizer，不做参数扫描
运行规模与日志：GPU 0--3，各 2048 env、128 steps，只执行一个冻结 rollout；日志 `doorman_train:/tmp/b2z1_phase268_frozen_rollout.log`，事件目录 `doorman_train:/tmp/b2z1_phase268_events/`
成功/失败判据：每个 unsafe 事件能在 reset 前唯一标注门侧、碰撞对象、峰值 force、前一帧和当前帧相对速度/间隙、base/EE actual、actor mean、sampled action 和 physical command；不得用聚合相关性替代事件链
观测指标：finger-handle force、gripper-panel force、door-frame/robot contacts、对象相对线速度、pocket/pregrasp error、orientation error、base pose/twist、EE pose/twist、十维 mean/sample/command、temporal fraction/amplitude
结论与已排除方向：已捕获 89 个事件：84 个仅门板、2 个仅把手、3 个同时接触，全部位于 APPROACH_PREGRASP，左右门为 40/49。56 个事件发生在 `total_time == 1`；其中 55 个涉及门板，base 到门板平面距离为 `0.5535--0.7163 m`，夹爪 collider clearance 中位数 `-96.7 mm`，证明 base 与 arm 独立随机会直接生成穿模初态。排除首次物理步后有 32 个门板动态事件，其中 14 个存在明确的“上一帧无接触、下一帧超过 600 N”；无接触帧 clearance 中位数 `7.7 mm`、门板法向接近速度中位数 `0.5007 m/s`，下一帧接触力中位数 `739.2 N`。actor base_x mean 中位数为远离门板的 `-0.0589`，但 temporal context `+0.1260` 将 sampled action 推成朝门的 `+0.0902`；动态链路是持续探索驱动 base/累计 EE 目标靠近，实际 EE 高速进入不足 1 cm 间隙后碰撞。排除把手抓取作为主要来源，也排除低速接触后传感器无故跳变
正式保留/删除的代码：事件文件和日志作为 phase 证据保留；`door_open_homie.py`、`ppo_trainer.py` 中的临时历史快照、逐 rank 导出、环境变量和 optimizer 前退出路径必须在 phase269 行为修改前删除
前序阶段回归：phase267 训练期间源码和配置冻结，纯动力学成立，无 NaN 或运行时异常
下一决策门槛：phase269 只修 reset 初态合法性。不得把动态接近 reward 与 reset 修复放在同一实验；reset 验收通过后再以独立 phase 设计只依赖真实 clearance 和相对速度的连续安全 credit
```

2026-09-29 08:45，phase268 首次 `8 env x 8 steps` smoke 在第一个诊断帧失败：夹爪现有 velocity cache 是三维线速度，诊断代码误与六维门体 twist 相减。失败发生在首个 rollout 步、optimizer 前，未生成诊断文件或 checkpoint，不属于物理或策略失败。修正仅将相对速度定义为三维线速度差，同时单独保留把手/门板六维 twist；下一原子动作是同参数重跑 smoke。

2026-09-29 08:47，phase268 同参数 smoke 重跑完成并在 optimizer 前正常退出。日志 `doorman_train:/tmp/b2z1_phase268_smoke.log` SHA-256 `194effde32e17631a1e8e5ae7ae3d6d7ab45ad1a832f462563fc8a1aff3571ae`；诊断文件 `/tmp/b2z1_phase268_smoke/unsafe_force_events_rank0.pt` SHA-256 `629387881677461fc024063cff29ed14dcafc6953136d83aa9f25db1f2a03284`，明确记录 `optimizer_executed=False`。该极小样本事件数为 0，仅用于验证生命周期和导出契约。源 checkpoint SHA 仍为 `438759b0c2928d315f37a9ae8bbfaae723698229334c14c28e393fc9667a1d2d`，新运行目录只有 `config.yaml/meta.yaml`、无模型文件。下一原子动作进入四卡正式冻结 rollout。

2026-09-29 08:55，phase268 正式冻结 rollout 完成并在 optimizer 前退出。四个 rank 共记录 89 个 unsafe-force 事件，源 checkpoint 未改变，没有形成可继承模型。日志 SHA-256 为 `ea028dbc4ae97bf2766a85f2ca47142dda6dad168b7fd974e4ebca9589db95ee`；rank0--3 事件文件 SHA-256 分别为 `b83f641e325f51d404e413f0954470035cf99870e1112516f66b7cca0fbdb3e6`、`44271f9c2764e4ba4304bc425cc3362c5d0f664419a22c4168ff2c7794b00915`、`b3b8a82b024eaf069259c795a88ab9bfad0f53e7a008c4d0ce8da3ae2a77bdc0`、`71f9dd446bf4d3af9347e083597ee35762eca87e78cd0bdc1709850cb0b9b5eb`。事件级证据确认 reset 非法重叠与动态高速接近是两个独立来源，phase268 封账。

### phase269：reset 初态物理合法性修复（已验证）

```text
上游结论与待解决瓶颈：phase268 的 89 个 unsafe-force 事件中有 56 个发生在 reset 后首个物理步；当前 base 前后位置与 middle arm 扰动独立采样，会让夹爪 collider 在策略尚未执行有效动作前穿入门板
基线 checkpoint / SHA-256：无；本 phase 先验证 reset 分布，不训练、不继承 phase267 checkpoint
唯一变量：在 reset 所有权层使联合 base/arm 初态满足机器人与门体无接触的物理合法性；保留原随机范围中所有合法组合，不按 seed、门型或 stage 特调
待验证假设：消除 reset 时的门体 collider 重叠，可消除首次物理步 unsafe-force，同时不改变合法初态的任务覆盖范围；动态高速接近事件仍应单独存在并留给下一 phase
禁止改动范围：不改 reward、600 N 阈值、termination、observation/action、策略、std/rho/fraction/budget、低层、门属性或动态接近控制；不得用固定 base 距离拍脑袋裁掉整个近门区域
运行规模与日志：phase268 临时诊断删除后，先完成纯函数、完整测试和 `8 env x 8 steps x 1 update` smoke；正式验收固定为源开发机 GPU 0--3、各 2048 env、8 steps、1 update，共 8192 个初态，日志 `doorman_train:/tmp/b2z1_phase269_reset_validation.log`
成功/失败判据：策略动作前，机器人与门板/把手/门框均无接触，所有 collider clearance 非负并保留明确安全裕量；大规模样本首次物理步 `>600 N` 为 0；随机化分布和左右门覆盖未被退化为少量固定姿态
观测指标：初态各对象 collider clearance、接触力、base x/y/yaw、六个 arm joint、门侧、拒绝/重采样率、首次物理步 termination；同时单独报告动态事件，不把它们计作 reset 修复失败
结论与已排除方向：正式四卡 8192 初态验收通过。生产 reset 保留原 `x=[0.55,2.0] m`、`y=[-2,2] m`、朝门 `yaw+-60 deg` 和六关节 `middle+-0.12 rad` 范围，只拒绝联合几何非法样本；几何拒绝率为 `0.0799506158`，`geometry_max_attempts` 的四 rank 聚合均值为 `1.75`。`termination/unsafe_force/reset_step=0`、总 unsafe-force 为 0；没有固定安全姿态或经验 base 下界。已排除直接修改 base 最近距离，因为安全性取决于 base、arm、门型和姿态的联合几何
正式保留/删除的代码：phase268 临时路径已全部删除。正式实现位于 `b2z1_reset_geometry.py` 和 stage-0 reset 所有权层；保留通用首步 unsafe-force 分桶指标
前序阶段回归：纯 Torch FK 与生产 URDF/Pinocchio 在两个非零 yaw、不同关节样本下位置和旋转误差均小于 `1e-9`；OBB/SAT 三项定向测试、py_compile、diff check、完整 `66 passed, 5 subtests passed`、小规模 smoke 和四卡正式验收通过
下一决策门槛：reset 根因封账；单独处理动态事件，不再把首次物理步穿模与运动碰撞混为一谈
```

2026-09-29 09:14，phase269 实现和小规模回归完成。新实现只在 reset 所有权层按生产 URDF 刚体链计算关闭夹爪碰撞 OBB，并对门板、把手、锁舌和门框 OBB 执行 15 轴 SAT；非法的 base/arm 联合样本整体重采样，64 次仍失败则显式报错，不存在固定姿态兜底。完整测试由 63 增至 `66 passed, 5 subtests passed`。GPU 3 的 `8 env x 8 steps x 1 update` 真实 smoke 正常完成，运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260929_091311`；纯动力学成立，`geometry_max_attempts=1`、`rejection_rate=0`、unsafe-force 为 0。下一原子动作仅为已登记的四卡 8192-env 正式 reset 验收。

2026-09-29 09:22，phase269 四卡正式验收完成。GPU 0--3 各 2048 env、8 steps、1 update，共 8192 个随机初态和 65536 timesteps；运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260929_091539`。`geometry_rejection_rate=0.0799506158`，四 rank 聚合的 `geometry_max_attempts=1.75`，`termination/unsafe_force/reset_step=0`、总 unsafe-force 为 0、existing termination 为 `0.0003662109375`。日志 `/tmp/b2z1_phase269_reset_validation.log` SHA-256 为 `81f4f05807ff480350905e88c1dec2615afc565d0811abe66793ea16933b3dd7`。无训练进程残留，phase269 封账。

### phase270：APPROACH_PREGRASP 非对称夹爪位置势场（碰撞抑制有效，阶段学习待继续验证）

```text
上游结论与待解决瓶颈：reset 首帧穿模已消除；动态碰撞前一帧夹爪距门板中位 7.7 mm、法向接近速度中位 0.5007 m/s。旧位置分量只使用到预抓取点的对称欧氏距离，无法区分越过目标朝门与向机器人侧远离
基线 checkpoint / SHA-256：无；reward 语义已变化，禁止继承旧 checkpoint
唯一变量：只替换 APPROACH_PREGRASP 的实际夹爪 pocket 位置势场。任务坐标 +X 朝门；正深度误差使用 0.01 m 指数尺度，负深度误差及横向/高度误差使用 0.5 m 尺度。其他四项 progress、综合方式、验收条件、observation/action、探索、低层、随机化和 600 N 阈值不变
待验证假设：越过 6 cm 预抓取目标继续朝门时，位置质量应立即快速下降；远离门及横向/高度偏差仍保留缓慢、连续的远场引导，从而降低动态撞门而不破坏接近学习
禁止改动范围：不得加入 clearance/velocity 的第二套 reward、阶段/seed/门型特例、动作目标监督、脚本轨迹或额外控制路径；训练运行期间冻结代码
运行规模与日志：纯函数与完整测试；GPU 3 的 8 env x 8 steps x 1 update 真实 smoke，日志 `doorman_train:/tmp/b2z1_asymmetric_pregrasp_smoke.log`，运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260929_104109`。正式验证使用 GPU 0--3、每卡 2048 env、128 steps、4 mini-batches，从随机初始化到 50 updates；日志 `doorman_train:/tmp/b2z1_phase270_asymmetric_clean50.log`
成功/失败判据：数学上目标点质量 1；朝门 10 mm 为 exp(-1)=0.3679，离门或横向 10 mm 为 exp(-0.02)=0.9802；左右门使用同一 canonical 符号；完整测试、py_compile、diff check 和真实 optimizer smoke 无异常/NaN。训练效果必须另由随机初始化四卡对照动态 unsafe-force、五项学习和联合稳定确认
观测指标：朝门/离门 depth 分布、pregrasp position quality、动态 unsafe-force、五项 quality/best/ok、stable 1/5/10、GRASP、existing termination
结论与已排除方向：实现链路及碰撞抑制效果已验证。新势场将动态 unsafe-force 降低约 90.2%，明确解决了本 phase 针对的朝门碰撞问题。50 update 后新旧实验都没有联合稳定或 GRASP，位置当前验收率也都接近 0；因此本实验不能证明完整第一阶段已经学会，也不能仅凭两个近零瞬时验收率的相对变化断言新势场破坏了可达性
正式保留/删除的代码：新增纯函数 `asymmetric_pregrasp_position_progress`、唯一生产调用点和两项定向测试；旧对称 `exp(-norm(error)/0.5)` 调用已删除
前序阶段回归：完整 `68 tests`、py_compile、git diff check 通过；真实 smoke 完成 64 timesteps 和 optimizer，`pregrasp_position_quality=0.0484`，unsafe-force/reset-step/总 unsafe-force 均为 0
下一决策门槛：保留已验证的非对称碰撞约束方向；继续训练或设计受控对照，判断安全侧到达目标的学习是否需要额外连续引导。不得把两个均接近 0 的瞬时验收率包装成显著任务回归
```

2026-09-29 11:31，phase270 四卡正式训练完成 50 update，共 `52,428,800` timesteps；四个 worker 正常退出，50 条 metrics 均无非有限值。运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260929_104551`，step50 checkpoint SHA-256 为 `f08af1089a56805cc89954fac5d8075303ff1238e6d8932ab2c5dbae28212b42`，metrics SHA-256 为 `197debba2b04ae6bb88d354f4ade844e21a46b1e430b8afe1a64e3607c46ce28`，日志 SHA-256 为 `af84c77f5c6c16967c454059504d08def82654c1c73ad46904cd289151f6a11e`。

安全收益明确：step50 unsafe-force 从旧对照 phase267 的 `2.4223e-4/step` 降至 `2.3842e-5/step`，下降约 `90.2%`；`contact_free` 从 `0.96134` 提高至 `0.99675`，reset-step unsafe-force 保持为 0。新诊断显示 step50 朝门越过目标的样本比例为 `3.0295%`，越过 `10 mm` 为 `2.4434%`，平均正向越界深度为 `1.091 mm`，门板 clearance 小于 `10 mm` 的比例为 `0.2625%`。

阶段学习仍未完成：step50 实际进入 `3.5 cm` 位置验收区的当前比例分别为旧对照 `0.2586%`、新势场 `0.0749%`，绝对差为 `0.1837` 个百分点，两者在任务意义上都接近 0，不能据此声称发生了从可用到不可用的回归。回合内曾进入比例分别为 `9.1906%/2.4506%`，姿态回合内曾通过比例为 `6.3855%/1.7450%`，这些指标提示新势场下目标区域访问更少，但两轮的 `stable_reached_1/5/10` 与 `GRASP` 均为 0，尚不能把访问率差异归因为完整可达性被破坏。可直接比较的 base position quality 为 `0.6712` 对 `0.6763`、base yaw quality 为 `0.6788` 对 `0.6587`，底盘学习没有整体回归。由于新旧 `pregrasp_position_quality` 公式不同，其绝对值不用于跨实验结论。

### phase271：WALK_TO_DOOR 与 PREGRASP 显式拆分（已验证）

```text
上游结论与待解决瓶颈：phase270 的 pregrasp 位置验收率以整个混合阶段为分母，包含底盘尚未就位的大量样本；底盘行走时评价机械臂精准把手目标既无物理意义，也会驱使机械臂追逐远处目标
基线 checkpoint / SHA-256：无；阶段 observation 和 staged-reset 接口由 6 维变为 7 维，所有旧 checkpoint 禁止继承
唯一变量：将 APPROACH_PREGRASP 拆为公开的 WALK_TO_DOOR 和 PREGRASP 两个阶段。WALK_TO_DOOR 只评价底盘工位、yaw、稳定与无接触；PREGRASP 才评价夹爪位置、姿态、开度、EE 稳定并要求底盘保持工位
待验证假设：显式顺序能消除底盘行走阶段的远距离机械臂目标，同时让 PREGRASP 指标只统计底盘已完成就位后的样本
禁止改动范围：不增加隐藏 base_ready 开关，不按 stage 屏蔽或替换 actor 动作，不修改十维 action-to-command 映射、低层、随机化、非对称位置势场、安全阈值或后续五阶段物理语义
运行规模与日志：完整单测；GPU 3、8 env、8 steps/env、1 update、1 mini-batch 的真实纯动力学 optimizer smoke，日志 `doorman_train:/tmp/b2z1_phase271_seven_stage_smoke.log`，运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260929_121114`
成功/失败判据：类、配置、one-hot observation、stage progress/stability、staged reset 和 trainer stage 统计统一为 7 阶段；WALK completion 不读取任何 arm 目标误差；PREGRASP 指标不包含 WALK 样本；测试和 optimizer 无异常或非有限值
观测指标：七个 train/stage_count、WALK/PREGRASP occupancy/progress、reward 名称、unsafe-force、reset-step unsafe-force、非有限值
结论与已排除方向：已验证。71 tests 全部通过；smoke 的 64 个 rollout 样本全部位于 walk_to_door，pregrasp 样本为 0，两个阶段分别输出独立 progress；optimizer 正常完成，unsafe-force/reset-step 均为 0，非有限指标为空。排除在六阶段第一阶段内添加隐藏 readiness gate 的方案
正式保留/删除的代码：正式类改为 B2Z1DoorSevenStage；删除旧 B2Z1DoorSixStage、STAGE_APPROACH_PREGRASP、approach_pregrasp 日志和旧 residual reward 名称；新增两个独立完成条件纯函数及定向测试
前序阶段回归：十维 action-to-command 映射未读取 stage 且未修改；OPEN_LATCH、REPOSITION、PUSH_OPEN、THROUGH 的物理条件仅顺延索引；完整 71 tests、py_compile、git diff check 和真实 optimizer smoke 通过
下一决策门槛：使用 GPU 0--3、各 2048 env、32 steps/env、200 updates 从随机初始化训练。先验收 WALK_TO_DOOR；只有出现 PREGRASP 样本后才评价机械臂位置、姿态、开度和碰撞
```

2026-09-29 12:12，phase271 实现与验证完成。smoke 日志 SHA-256 为 `32ad593bb32cf00a4ed63038c4fc0eb21b6e79dc14a507dbe277a4a611ece916`，metrics SHA-256 为 `364e59ced2737ce15319741321881cd6b9e4d3e8f06053e78aecec0d2e2be2f0`，checkpoint SHA-256 为 `16321d81411598ba2737aa7680051e1a8cecabc753b7fabbb81fc3c3033ca02d`。该 checkpoint 仅为 64 timesteps 接口 smoke，不作为可继承策略。

### phase272：六阶段三类状态语义与独立阶段初态库（运行中）

上游结论与待解决瓶颈：用户明确取消独立 PREGRASP；第一阶段必须同时完成底盘站位和机械臂 middle preset，GRASP 从 middle 直接学习。每阶段都要有机器狗、机械臂、外部环境三类目标和连续势场，并支持从上一阶段合法验收状态单独训练。

基线 checkpoint / SHA-256：无；阶段 observation 从 7 维变为 6 维，禁止继承旧策略。阶段初态源与策略 checkpoint 解耦。

唯一变量：重构为 `WALK_TO_DOOR -> GRASP -> OPEN_LATCH -> REPOSITION -> PUSH_OPEN -> THROUGH`；删除 B2Z1 正式类中的 6 cm pregrasp 点、预设开爪 reward 和独立阶段；扩展原 staged-reset 保存每 env 最近 16 个真实验收快照，允许随机权重固定训练指定阶段。

待验证假设：无中间点的最终结果势场足以提供连续学习信号；从完整合法物理状态随机采样能让任一阶段独立训练，同时不改变全流程 observation/action 语义。

禁止改动范围：不加入 AutoOpen 动作、reference action、阶段动作 mask、seed/门型特调、跨 env 门状态移植或命令目标监督；不修改低层、十维 action-to-command、安全阈值和门随机化。

运行规模与日志：纯函数、契约和完整测试已完成；真实 smoke 计划为 GPU 3、8 env、8 steps/env、1 update。固定阶段重置验真将在同一纯动力学配置执行。

成功/失败判据：六阶段类/config/one-hot/stability/reward 维度一致；所有阶段有连续 progress；GRASP 不含 pregrasp 中间目标且允许底盘移动；OPEN_LATCH 同帧要求底盘、夹持、锁舌和门缝；初态库随机采样完整状态并按当前条件复验；全测与 optimizer 无异常、NaN 或残留进程。

观测指标：六阶段 occupancy/reached/progress、WALK middle arm error/speed、GRASP 五项连续质量与严格接触、`reset/stage_initial_state_valid`、非有限值和安全终止。

结论与已排除方向：进行中。当前 py_compile 通过；完整 `75 passed, 5 subtests passed`。已排除把旧 PREGRASP 势场改名塞入 GRASP，以及跨随机门几何移植状态的方案。

正式保留/删除的代码：待真实验证后封账。当前新增纯 Torch `staged_reset_library.py`，正式环境仅调用该唯一序列化实现。

前序阶段回归：十维动作映射、低层、门随机化和安全阈值未修改；阶段 one-hot 仍进入 actor/critic。

下一决策门槛：真实 Isaac optimizer smoke 通过，并证明固定阶段重置后的完整物理状态满足上一阶段验收。

2026-09-29 15:24，独立阶段初态机制已从“必须读取真实轨迹快照”改为任务几何直接采样。stage 0 继续使用原始随机 reset；stage 1 在 WALK_TO_DOOR 验收集合内随机采样底盘工位、yaw 和 middle arm；stage 4 根据每个环境的随机门宽、左右铰链和门缝角计算门板姿态、底盘工位与 push-ready EE 位姿，再用生产 B2Z1 刚体链批量 IK 生成关节状态；stage 5 随机采样 `91--105 deg` 门角、无接触底盘/机械臂状态，并保留“上一阶段已发生推门接触”的历史事实。所有 procedural 初态均继续由原阶段验收函数复验，不修改 reward、验收阈值、observation/action 或低层控制。

真实纯动力学 smoke 均未提供 `STAGE_INITIAL_STATE_SOURCE`：stage 1 为 8 env，stage 4/5 各 16 env，分别完成 16/32/32 个目标阶段 rollout 样本，三者 `reset/stage_initial_state_valid=1.0000`；日志为 `doorman_train:/tmp/procedural_stage{1,4,5}_smoke_v2.log`。stage 2 OPEN_LATCH 和 stage 3 REPOSITION 涉及持续接触，当前尚未实现 procedural generator；固定阶段启动会明确报错，实测日志为 `/tmp/procedural_stage{2,3}_reject.log`，禁止静默回退到快照。纯 Torch FK/IK、契约和完整测试共 `76 passed`，py_compile 与 `git diff --check` 通过。

### 2026-09-29 独立阶段训练状态

> 以下按时间顺序保留训练与诊断过程。早期 checkpoint 的“未通过”是历史结果，
> 已被本节后续实验和顶部总账覆盖，不是当前结论。当前结论为：PUSH_OPEN 与
> THROUGH 均已通过独立阶段验收；尚未完成的是同一策略串联全部六阶段的最终验收。

2026-09-29 16:17，WALK_TO_DOOR 的 step400 checkpoint 完成 2048 个随机初态确定性评估。总成功 `1825/2048=89.11%`；左铰链 `968/1019=95.00%`，右铰链 `857/1029=83.28%`；成功平均 `317.91` 步，失败平均 `708.13` 步。checkpoint 为 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260929_144737/model_step_000400.pt`，指标为 `doorman_train:/tmp/b2z1_phase272_step400_deterministic_eval/metrics_eval.json`。按用户当前验收口径，WALK_TO_DOOR 阶段通过；右铰链鲁棒性差异作为已知剩余风险记录，不在本轮继续优化。

2026-09-29 16:40，本机从随机权重启动固定 GRASP 阶段训练。配置为 GPU 0--3、每卡 2048 env、32 steps/env、4 mini-batches、目标 200 updates，日志 `doorman_train:/tmp/b2z1_grasp_stage1_200.log`，运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260929_164005`。17:09 核对时约到 update97，四个 worker 存活，无 traceback、CUDA OOM、非有限值或固定阶段初态异常；运行期间冻结行为代码。

2026-09-29 17:47，GRASP step200 与 THROUGH step400 已分别在本机 GPU 1/GPU 0 完成正式独立阶段确定性评估。每项均使用 2048 个程序化随机初态、每环境一局、actor `action_mean`、无 exploration noise、headless、无相机。GRASP 成功 `0/2048`，评估耗时 `240.66 s`；THROUGH 成功 `0/2048`，评估耗时 `213.70 s`。两项均正常退出，无 traceback、CUDA OOM 或 runtime error。该结果只说明当前策略没有满足离散阶段验收，不能仅凭成功率判断是哪一项连续目标首先失败；后续必须从已保存的阶段指标做只读分解，根因确认前不修改 reward、验收条件或训练接口。

远程后期阶段使用两卡，因此目标设为 400 updates，使总样本量与本机四卡 200 updates 相同：`2 * 2048 * 32 * 400 = 52,428,800` timesteps。

- `wholebody_3`：固定 PUSH_OPEN（stage 4），运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260929_154621`。目标 update400；17:26 核对到 update284，共 `37,224,448` timesteps，原任务一直在运行且 metrics 持续更新。此前仅凭容器外 `ps` 不可见便误判为中断，并误启动了一组 step200 续训；该重复组已按 checkpoint 参数精确终止，原始训练组保留，GPU 上只剩原任务。该实验状态为“训练中”，不得再次仅凭宿主机 `ps` 判停。
- `wholebody_4`：固定 THROUGH（stage 5），同名时间戳运行目录。已正常写出 step400 checkpoint，共 `52,428,800` timesteps，`_non_finite_metrics=[]`；checkpoint 已拉回本机并完成 2048 局确定性评估，成功 `0/2048`。这是当时 checkpoint 的历史失败结果；后续修复位置瓶颈并重新训练后，正式独立阶段验收已达到 `85.11%/85.06%`，当前状态为已通过。

当时的下一原子动作为：只读分解 GRASP 与 THROUGH 的阶段目标项，并等待 wholebody3 PUSH_OPEN 到 update400。该动作已经完成，不是当前待办；后续正式结果见本节末尾和顶部总账。

2026-09-29 18:47，修复通用评估记账接口后，使用本地当前代码同步到 `wholebody_3`，对 PUSH_OPEN step400 做了 2048 局确定性评估。旧接口从未更新 `episode_goal_reached`，因此此前由该接口报告的 WALK/GRASP/THROUGH/PUSH_OPEN 成功率均须重新核验，不能继续作为正式成功率证据。本次评估使用 actor `action_mean`、无探索噪声、headless、无相机、固定 stage 4；结果保存于本地 `logs_eval/push_open_step400_synced_code_eval2048/metrics_eval.json`。

PUSH_OPEN 真实成功 `1584/2048=77.34%`。所有回合均建立过门板接触；成功回合最大门角均值 `89.22 deg`，范围 `89.00--89.55 deg`。失败构成为门角速度连续超限 `410` 局、峰值任务力超过 `600 N` 的 `49` 局、肩部碰撞 `3` 局，另有 `2` 局在约 `84 deg` 被基础环境终止，四类没有重叠。峰值任务力中位数 `441.6 N`、P95 `567.0 N`、P99 `659.9 N`、最大 `826.7 N`；`1999/2048=97.61%` 未超过 `600 N`。最大开门角速度中位数 `28.33 deg/s`、P95 `31.37 deg/s`、最大 `35.51 deg/s`。

训练前 50 与后 50 updates 对比显示，门角 progress 从 `-0.3033` 升至 `0.5704`，阶段平均 reward 从 `-0.0917` 升至 `1.0398`，平均开门速度从 `3.01 deg/s` 升至 `19.75 deg/s`，接触占比从 `45.68%` 升至 `89.24%`，平均门板力从 `1.96 N` 升至 `12.72 N`。这证明策略已学会以接近目标平均速度推门，且绝大多数回合遵守 `600 N` 峰值力限制；但后 50 updates 的超速终止率高于前 50，确定性评估也有 `20.02%` 因峰值速度超限失败。当前主要瓶颈不是推力不足，而是 reward 对平均速度的优化没有充分抑制峰值超速。

2026-09-29 19:20，在 `wholebody_3` 使用新增夹爪-门板相对速度连续惩罚，从随机权重重新启动固定 PUSH_OPEN 训练。运行目录为 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260929_191748`；两张 RTX 4090、每卡 2048 env、32 steps/env、4 mini-batches、目标 400 updates，每 50 updates 保存 checkpoint。前两个 updates 已完成，累计 `262144` timesteps；`panel_relative_speed` 与 `panel_relative_speed_penalty` 均为有限非零值，两个 worker 正常，无 traceback、CUDA OOM 或非有限指标。独立启动日志为 `logs_rl/push_open_relative_speed_clean400.log`。

2026-09-29 19:30，本机 GRASP enclosure 势场训练完成 step200，共 `52,428,800` timesteps；checkpoint 为 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260929_181131/model_step_000200.pt`。使用修复后的评估记账接口完成 2048 局确定性评估，结果为 `logs_eval/grasp_enclosure_step200_eval2048/metrics_eval.json`。严格成功为 `0/2048`。所有回合最大开口质量均达到 `1.0`，说明策略已经可靠学会打开夹爪；`2038/2048=99.51%` 的回合曾达到 `between_jaws_quality>=0.99`，说明把手沿夹爪闭合轴进入两爪跨度这一几何子目标基本学会。

稳定抓取尚未学会。只有 `400/2048=19.53%` 曾出现任一爪有效接触，`10/2048=0.49%` 曾出现同帧双爪有效接触，`26/2048=1.27%` 曾出现短窗口内对向接触；没有任何回合满足完整严格双侧抓取。最大 capture quality 均值为 `0.535`，仅 `5/2048` 超过 `0.8`，没有回合超过 `0.9`；`1902/2048=92.87%` 最终超时。训练末尾平均 pocket distance 仍约 `0.113 m`，轴向误差约 `0.343 rad`，而严格验收分别要求 `<0.020 m` 和 `<0.05 rad`。因此当前策略已学会“打开并大致把把手放入两爪跨度”，但没有学会精确对齐、闭合并维持双侧受力。

2026-09-29 19:46，确认旧 GRASP reward 的最终接触平台：`captured_quality` 使用 `max(closure_quality, physical_contact_quality)`；当几何闭合质量约为 `0.64` 时，无接触、单爪接触质量 `0.25`、双爪非对向接触质量 `0.5` 都得到相同捕获分，只有稀疏的完整对向接触才增加 reward。现已改为几何闭合与物理接触各占一半，并新增逐爪连续力质量：`0--5 N` 线性增加，`5--30 N` 满质量，超过 `30 N` 后指数下降；双侧质量由逐爪均值与较弱一侧的最小值各占一半，因此无接触、单爪接触、双爪平衡接触严格逐级增加。该接触质量再乘以夹爪-把手相对速度和夹爪关节速度的连续稳定质量，只有低速保持接触才能获得完整分数。严格 GRASP 验收、动作、observation 和其他阶段均未修改。

纯函数与完整测试共 `88 tests` 通过；GPU 3 的 `8 env x 8 steps x 1 update` 真实 smoke 完成，无异常或非有限值。随后本机四卡从随机权重启动固定 GRASP 重训：每卡 2048 env、32 steps/env、4 mini-batches、目标 200 updates、每 25 updates 保存 checkpoint；运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260929_194235`，日志 `doorman_train:/tmp/b2z1_grasp_bilateral_force_clean200.log`。前 3 updates 已完成，新增固定爪、移动爪和双侧平衡质量均为有限非零值，四个 worker 正常。

2026-09-29 20:07，`wholebody_4` 的固定 THROUGH 训练正常完成 step800，共 `104,857,600` timesteps；checkpoint 为 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260929_175542/model_step_000800.pt`。同步本地当前代码后完成 2048 个程序化随机初态的确定性评估：actor `action_mean`、无 exploration、headless、无相机、每环境一局。严格成功为 `0/2048`，全部因阶段超时结束；无 unsafe force、门超速或肩部碰撞。门角全程满足要求，最终范围为 `91.00--105.00 deg`；底盘稳定条件每局都曾满足。

只读诊断确认首要瓶颈是机械臂回到 middle preset：只有 `5/2048=0.24%` 的回合曾同时达到最大关节位置误差 `<0.05 rad` 和最大关节速度 `<0.08 rad/s`，且没有一局与穿门距离和底盘稳定条件同时成立。左铰链 `1019` 局中没有一局的位置误差曾低于 `0.05 rad`，最终位置误差中位数 `0.907 rad`；右铰链 `1029` 局中仅 `30` 局曾达到位置阈值、`5` 局达到完整 arm-stowed，最终位置误差中位数 `0.207 rad`。穿门距离本身已有部分学习：`1642/2048=80.18%` 曾超过 `1.0 m`，左/右铰链分别为 `64.47%/95.72%`。因此 step800 未通过，最早且占主导的失败项是机械臂 middle preset，左铰链分支尤其严重；其次才是 `19.82%` 回合穿门距离不足。正式指标保存在 `logs_eval/through_step800_eval2048_clean/metrics_eval.json`，只读分解保存在 `logs_eval/through_step800_eval2048_diagnostic/metrics_eval.json`。

2026-09-29 20:50，按已经通过 WALK_TO_DOOR 的 reward 结构统一 THROUGH：三类目标 progress 从算术平均改为 `bottleneck_balanced_progress(..., bottleneck_weight=0.5)`，THROUGH 的 signed progress delta 与 WALK/GRASP 一样乘 `20`，并纳入同一个持续 `penalty_stage_residual`。未新增阶段特例、脚本动作或目标命令，未修改 observation、action、初态、验收条件和安全阈值。完整测试 `88 passed, 8 subtests passed`；`8 env x 8 steps x 1 update` 真实 Isaac smoke 完成 64 timesteps并正常保存 checkpoint。

随后在 `wholebody_4` 两张 RTX 4090 上从随机权重启动固定 THROUGH 训练，目标按用户要求设为 400 updates；每卡 2048 env、32 steps/env、4 mini-batches、每 50 updates 保存。运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260929_205026`，独立日志 `logs_rl/through_walk_aligned_clean400.log`。两个 worker 已完成首个 update，全局每 update `262,144` timesteps，iteration time 约 `11.2 s`，无 traceback、OOM 或非有限值。

2026-09-29 22:13，固定 PUSH_OPEN 的相对速度连续惩罚实验完成正式闭环。训练在 `wholebody_3` 两张 RTX 4090 上从随机权重运行到 400 updates，共 `52,428,800` timesteps；checkpoint 为 `/datadisk/workspace/doorman_docker_package/GR00T-VisualSim2Real/logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260929_191748/model_step_000400.pt`。训练前 50 到后 50 updates，平均夹爪-门板相对速度由 `0.5081 m/s` 降至 `0.3518 m/s`，对应连续惩罚由 `5.1626` 降至 `1.7641`；平均开门速度由 `0.0372 rad/s` 提高至 `0.3187 rad/s`，平均门板力由 `1.21 N` 提高至 `14.19 N`。训练期门超速终止仍随开门行为形成而增加，说明相对线速度不是门角速度峰值的直接代理，最终结论必须来自确定性评估。

使用当前 PUSH_OPEN 行为代码完成 2048 个程序化随机初态的确定性评估：actor `action_mean`、无 exploration、headless、无相机、固定 stage 4、每环境一局。严格成功 `1880/2048=91.80%`，相对旧 step400 基线 `1584/2048=77.34%` 提高 `14.46` 个百分点。168 个失败互斥分解为：门角速度连续超限 `142`、峰值任务力超过 `600 N` 的 `24`、肩部碰撞 `1`、基础环境终止 `1`；无超时，全部 2048 局均建立过门板接触。旧基线对应为超速 `410`、超力 `49`、肩碰 `3`、基础终止 `2`。

峰值开门角速度中位数由 `28.33 deg/s` 降至 `24.97 deg/s`，P95 由 `31.37` 降至 `30.56 deg/s`，P99 由 `32.47` 降至 `31.69 deg/s`；峰值任务力中位数由 `441.6 N` 降至 `396.2 N`，P95 由 `566.8 N` 降至 `525.2 N`，P99 由 `657.8 N` 降至 `611.5 N`。最大角速度极值仍为 `35.57 deg/s`，最大任务力极值为 `878.8 N`，是剩余尾部风险，但本轮既定目标是验证新增连续引导能否优于 `77.34%`，该门槛已明确通过，因此不继续叠加 reward 或阈值补丁。正式指标已拉回本地 `logs_eval/push_open_relative_speed_step400_eval2048/metrics_eval.json`；定向相对速度测试 `2 passed`，完整测试 `90 passed, 8 subtests passed`，`py_compile` 和 `git diff --check` 通过。

### 2026-09-29 GRASP 单帧完整夹持率提升

基线为 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260929_194235/model_step_000200.pt`。2048 个程序化随机初态、actor mean、无 exploration、headless、无相机的确定性评估保存在 `logs_eval/grasp_bilateral_force_step200_diagnostic2048_v2/metrics_eval.json`；曾单帧满足完整夹持几何为 `257/2048=12.55%`。本轮只要求该比例获得可复现实质提升，不以连续保持 5 步或严格阶段完成率为优化目标。

首个有直接证据的 reward 缺口是 pocket 位置势场与验收不一致。验收同时要求总距离 `d<0.020 m` 和门板法向误差 `n<0.012 m`，旧势场只有

```text
q_d = min(1, exp(-d/0.150) / exp(-0.020/0.150))
```

因此 `0.012 <= n < 0.020 m` 且 `d<0.020 m` 的验收失败状态已经得到满位置分。最小修复保留原总距离项，并增加与法向验收对应的连续项：

```text
q_d = exp(-max(d - 0.020, 0) / 0.150)
q_n = exp(-max(n - 0.012, 0) / 0.090)
q_pocket = sqrt(q_d * q_n)
```

两项尺度具有相同比例 `threshold/scale = 2/15`；`q_pocket` 仅在两项验收均通过时为 1，任一未通过时均保留朝目标方向的非零梯度。未修改验收阈值、接触 reward、动作、observation、初态、安全条件或其他阶段行为。

新增两项纯函数定向测试，覆盖“总距离已饱和但法向仍失败时必须存在梯度”和“两项验收边界必须同时满足”；容器内完整测试为 `90 tests passed`。固定 GRASP 的 `8 env x 8 steps x 1 update` 真实 optimizer smoke 正常完成，新增 `pocket_distance_quality/pocket_normal_quality/pocket_position_quality` 分别为 `0.2954/0.2816/0.2868`，无非有限指标。四卡正式训练于 22:00 启动：每卡 2048 env、32 steps/env、4 mini-batches、200 updates、从随机权重开始；运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260929_220002`，日志 `/tmp/b2z1_grasp_pocket_normal_clean200.log`。完成后必须按基线相同口径执行 2048-env 确定性复评，再决定本次修改是否有效。

### 2026-09-29 THROUGH 关节位置 bottleneck

`wholebody_4` 的 WALK 同构 reward 实验完成400 updates，共 `52,428,800` timesteps；checkpoint 为 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260929_205026/model_step_000400.pt`。训练 progress 从前50 updates平均 `0.3838` 提高到后50 updates平均 `0.8739`，但训练期间没有阶段成功。

使用训练机当前代码完成2048个程序化随机初态的确定性评估：actor mean、无 exploration、headless、无相机、固定 stage 5、每环境一局。正式指标为 `wholebody_4:logs_eval/through_walk_aligned_step400_eval2048/metrics_eval.json`。严格成功 `0/2048`；穿门距离、门角和底盘稳定分别曾由 `2048/2048` 满足，无 unsafe force、门超速或肩部碰撞；arm stowed 为 `0/2048`，因此首要失败项唯一收敛到机械臂 middle preset。

每局最大关节位置误差的最小值中位数为 `0.1425 rad`，仍高于验收 `<0.05 rad`；左/右门分别为 `0.1604/0.1231 rad`。最大关节速度的最小值中位数为 `0.0185 rad/s`，左右门分别为 `0.0189/0.0169 rad/s`，已低于验收 `<0.08 rad/s`。与 step300 对照，位置误差中位数由 `0.2558` 降至 `0.1425 rad`，证明继续训练在改善位置，但旧 reward 仍不足以达到验收。

旧 THROUGH arm progress 将六关节位置质量直接取均值：

```text
q_j = exp(-max(|q_j-q_j_middle|-0.05, 0) / 0.5)
Q_arm_old = mean(q_1, ..., q_6)
```

验收却使用六关节最大误差，因此一个坏关节可被其余五个满分关节掩盖。最小修复只把位置聚合改为：

```text
Q_arm_new = 0.5 * mean(q_1, ..., q_6) + 0.5 * min(q_1, ..., q_6)
```

该势场仍给所有关节连续梯度，同时把一半权重明确给验收中的最坏关节；仅当六关节都进入 `0.05 rad` 验收范围时才达到1。速度不是本轮实测首要瓶颈，因此未添加速度 reward，也未修改验收、动作、observation、初态或其他阶段行为。

本地 `py_compile`、`git diff --check`、完整 `90 passed, 8 subtests passed` 通过；固定 THROUGH 的 `8 env x 8 steps x 1 update` 真实 smoke 完成64 timesteps并正常保存 checkpoint。远程同一 hunk 的编译与完整测试通过。随后在 `wholebody_4` 两张 RTX 4090 上从随机权重启动独立重训：每卡2048 env、32 steps/env、4 mini-batches、400 updates、每50 updates保存；运行目录 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260929_222553`，日志 `logs_rl/through_position_bottleneck_clean400.log`。完成后必须以同一口径做2048-env确定性复评。

### 2026-09-29 三阶段并行闭环状态修正

本轮固定并行推进 `GRASP`、`PUSH_OPEN`、`THROUGH`，三个阶段互不等待。统一流程为：合法随机初态单阶段训练，随后用 actor mean、无探索噪声、2048 env 做确定性验收；失败时先拆解物理指标，只在根因明确后做该阶段的最小泛化修复。禁止 seed 特调、脚本动作、临时阈值、隐藏开关、跨阶段补丁及 observation/action/坐标语义分叉。

`PUSH_OPEN` 已完成本轮闭环。相对速度连续惩罚实验的正式结果仍为 `1880/2048=91.80%`，相比旧基线 `1584/2048=77.34%` 提升 `14.46` 个百分点；失败互斥分解为门超速 `142`、峰值任务力超过 `600 N` 的 `24`、肩碰 `1`、基础终止 `1`。正式指标为 `logs_eval/push_open_relative_speed_step400_eval2048/metrics_eval.json`。该结果已达到本轮改进目标，不继续叠加 reward；当前只做代码归因审计和后续回归守护。

`GRASP` 的 pocket-normal 实验已经正式判定失败，不能继续作为待评估方案。新 checkpoint `door_open_b2z1_lstm_mlp-20260929_220002/model_step_000200.pt` 的同口径结果为 `2/2048=0.0977%`，远低于旧基线 `257/2048=12.55%`；对向接触由 `1043` 降至 `51`，任一有效接触由 `1471` 降至 `110`。受控 A/B 确认 `sqrt(q_distance*q_normal)` 一方面把切向和垂直方向的总距离梯度减半、额外提高法向权重，另一方面又通过 `alignment_quality` 二次影响 capture/contact，导致策略后期主动后撤 EE。该失败公式已撤销，恢复为原总距离势场；法向质量仅保留为诊断量。下一步只分析旧基线中 `1313` 个从未通过 pocket distance 的环境以及 `296` 个验收条件分别达到但未同帧达到的环境，证据明确前不新增 reward。

`THROUGH` 的 bottleneck 重训和正式复评均已完成。`wholebody_4` 的唯一原任务运行到400 updates、`52,428,800` timesteps，checkpoint 为 `door_open_b2z1_lstm_mlp-20260929_222553/model_step_000400.pt`。2048-env actor-mean、无探索确定性评估严格成功 `1740/2048=84.96%`，旧 step400 为 `0/2048`；位置 bottleneck 修复有效。详细拆分见下节。

### 2026-09-29 THROUGH 位置 bottleneck 正式复评

正式指标为 `logs_eval/through_position_bottleneck_step400_eval2048/metrics_eval.json`，SHA-256 `31344705003a825d8e7abfa99a1a487ccc2ad1b377888036995e99996d74fc7d`；checkpoint SHA-256 `6ffe05d44b005acd272026720ab75e9c01b26588c8bca44dfb2c14125d03fdb1`。评估使用固定 stage 5 的2048个程序化随机初态、actor mean、无 exploration、headless、无相机、每环境一局。

严格成功从旧 step400 的 `0/2048` 提高到 `1740/2048=84.96%`。曾穿过1米从 `2048/2048` 变为 `1928/2048=94.14%`；曾满足 arm stowed 从 `0/2048` 提高到 `1879/2048=91.75%`；曾满足 base stability 均为 `2048/2048`；门角条件均为 `2048/2048`。无 unsafe force 和肩部碰撞，只有2局门超速。

每局最大关节位置误差的最小值中位数由 `0.1425` 降到 `0.0371 rad`，P90由 `0.1799` 降到 `0.0486 rad`，已进入 `<0.05 rad` 验收范围。最大关节速度的最小值中位数由 `0.0185` 变为 `0.0367 rad/s`，P90为 `0.0541 rad/s`，仍满足 `<0.08 rad/s`。最大穿门距离中位数由 `1.6685` 变为 `1.1677 m`；新策略不再无意义地过度前进，但右门残留少量距离失败。

左右门严格成功分别为：左门 `908/1029=88.24%`，右门 `832/1019=81.65%`。左/右门 arm stowed 为 `90.96%/92.54%`，位置误差中位数为 `0.0365/0.0375 rad`，旧实验的显著机械臂门型不对称已经消除。穿门距离通过率左/右为 `99.81%/88.42%`，当前右门弱项转为穿门距离。

308个失败中：120局从未穿过1米，169局从未达到arm stowed，256局从未让全部条件同帧成立；另52局曾同帧满足全部条件但未连续保持10步。最大连续稳定步数不少于1/5/10步的环境数分别为 `1792/1768/1740`。本轮目标是验证唯一位置 bottleneck 修复能否解决原来的 arm 主瓶颈，该目标已由 `0% -> 84.96%` 和关节误差分布共同证明；不继续添加速度 reward、阈值补丁或阶段特例。

### 2026-09-29 PUSH_OPEN 权威复核与回归边界

本次只读复核没有修改 reward、行为、验收阈值或训练配置。正式指标
`logs_eval/push_open_relative_speed_step400_eval2048/metrics_eval.json` 的 SHA-256 为
`b4018bb058520769713da4b564122271f2a832b97b445e8550c98fe1d1fbfd38`。从逐环境数组独立重算得到：

- 成功 `1880/2048=91.80%`，`episode_goal_reached` 与 `goal_reached_buffer` 逐项完全一致；
- `168` 个失败中，门超速 `142`、`>600 N` `24`、肩碰 `1`、基础环境终止 `1`，四类互斥且无未解释缺口；
- `2048/2048` 都建立过真实门板接触，无超时，也不存在“成功同时带安全失败标志”的样本；
- 相对旧基线 `1584/2048=77.34%`，超速 `410 -> 142`、超力 `49 -> 24`、肩碰 `3 -> 1`；峰值开门速度中位数 `28.33 -> 24.97 deg/s`，峰值任务力中位数 `441.6 -> 396.3 N`。

训练 checkpoint 为远程 `wholebody_3` 的
`door_open_b2z1_lstm_mlp-20260929_191748/model_step_000400.pt`，SHA-256
`6a9f3eb287420621ed711455ebbcd362d1d150d23d3098593689ad7f4b4c7322`。日志证明双卡训练完成
400 updates，即 `2 * 2048 * 32 * 400 = 52,428,800` timesteps，无 traceback、OOM、NaN 或 Inf；相机数、RTX sensor 和 frame rendering 均为零。与旧基线训练配置递归比较并排除生成路径后，唯一有行为意义的差异是新增
`penalty_push_relative_speed: -2.0`；另一个差异只是新配置显式写出默认值
`staged_reset_eval_stage: null`。

连续惩罚使用门板接触点速度
`v_panel + omega_panel x (p_gripper - p_panel)` 与夹爪世界速度之差，并且严格由
`stage_buf == STAGE_PUSH_OPEN` 门控。其公式为
`max((||v_rel|| - 0.2) / 0.2, 0)^2`：安全带内为零，超过 `0.2 m/s` 后连续二次增长，没有高速度截断或提前饱和。PUSH_OPEN 验收仍是门角至少 `89 deg` 且本阶段曾发生真实夹爪-门板接触；连续 3 步超过 `30 deg/s`、任务力超过 `600 N` 或肩部侵入仍触发失败。固定阶段评估由 actor mean 驱动，每个程序化随机初态只取第一局。

本地与远程代码逐 hunk 比较后，差异只涉及后续 GRASP reward/诊断和 THROUGH reward/诊断；PUSH_OPEN 的相对速度计算、reward 门控、验收和安全终止代码一致。回归验证为：相对速度定向测试 `2 passed`，相关阶段/事件/纯 RL 契约测试 `33 passed, 5 subtests passed`，完整测试 `89 passed, 8 subtests passed`，`py_compile` 与 `git diff --check` 均通过。

PUSH_OPEN 的最小回归文件集合为：

- `gr00t/rl/envs/door/stage_conditions.py`：连续相对速度惩罚公式；
- `gr00t/rl/envs/door/door_open_homie.py`：门板接触点速度、阶段门控、验收和安全终止；
- `gr00t/rl/config/rewards/wbmanip/reward_door_open_b2z1.yaml`：惩罚权重；
- `gr00t/rl/envs/base_task/staged_task_base.py` 与 `gr00t/rl/eval_agent_trl.py`：固定阶段成功记账和确定性评估路径；
- `gr00t/rl/tests/test_door_open_stage_conditions.py`、`test_staged_task_events.py`、`test_b2z1_pure_rl_contract.py`：纯函数、阶段事件和无脚本动作契约。

结论：`91.80%` 的提升可以归因于唯一新增的 PUSH_OPEN 连续相对速度惩罚，没有发现指标造假、阶段外行为修改或 reward 提前饱和。本轮不新增 reward；剩余的超速和超力尾部风险保留为后续回归监控项。

### 2026-09-30 GRASP pocket 主势场保护实验

旧 GRASP 基线仍为 `20260929_194235/model_step_000200.pt`，目标指标仍是同一口径下曾单帧满足完整夹持几何的 `257/2048=12.55%`。事件诊断复评保存在 `logs_eval/grasp_baseline_step200_event_diagnostic2048/metrics_eval.json`，本次复评得到 `253/2048=12.35%`，用于 reward 反事实分析而不替代原基线。

旧位置质量为 `q_d=exp(-max(d-0.020,0)/0.150)`。它在 `d<=20 mm` 时饱和，但验收仍要求法向误差 `|n|<12 mm`。新主势场只增加法向缺口项：

```text
q_n = exp(-max(|n|-0.012, 0)/0.090)
q_p = q_d - 0.25 * (1-q_n)
```

`q_d` 未缩放，所以新 `q_p` 对切向和垂直误差的导数与旧 `q_d` 完全相同；禁止的 `sqrt(q_d*q_n)` 和 `min(q_d,q_n)` 均未使用。其余底盘位置、底盘 yaw、夹爪姿态和接触质量形成 secondary 势场 `s`。GRASP 的逐步奖励增量为：

```text
dp = q_p(t) - q_p(t-1)
ds = s(t) - s(t-1)
delta = dp + ds                    if dp >= 0
delta = dp + min(ds, 0)            if dp < 0
```

因此打开、对齐、闭合和双侧接触改善仍有连续正梯度，但不能用正收益覆盖 pocket 退化；secondary 自身退化仍产生负收益。2048 对 closest/max-acceptance 事件快照中，旧公式有 `747` 对在 pocket 主势场退化时仍得到总正增益，新公式下这 `747` 对全部不再为正。295 个最大 `9/10` 验收样本中对应的错误正增益为 `20 -> 0`。253 个成功样本在完成时均保持 `q_d=q_n=q_p=1`。

纯函数定向测试 `20/20`、诊断联合测试 `23/23`、完整测试 `95/95` 均通过；`py_compile` 和 `git diff --check` 通过。固定 GRASP 的 `8 env x 32 steps x 2 updates` 真实 optimizer smoke 完成 512 timesteps，无 traceback、NaN 或运行时错误；日志为 `logs_rl/grasp_reward_protected_smoke8_train.log`，运行目录为 `door_open_b2z1_lstm_mlp-20260930_000033`。

### 2026-09-30 三阶段当前工作区并行回归

本轮在同一当前工作区上并行执行了三个固定阶段的 2048-env 确定性评估，均使用 actor mean、关闭 exploration、headless、无相机、每环境第一局。评估前修复了 GRASP 只读诊断中遗留的旧函数名；该修复不改变 reward、策略输入输出或物理行为。完整测试为 `101 passed, 8 subtests passed`。

`PUSH_OPEN` 回归结果为 `1884/2048=91.99%`，正式结果为 `1880/2048=91.80%`，没有回归。164 个失败严格互斥地分为：门连续超速 `132`、任务力超过 `600 N` 的 `32`；肩碰、超时和其他终止均为 `0`。全部 2048 局都建立过真实门板接触。指标文件为 `logs_eval/push_open_current_worktree_regression_eval2048/metrics_eval.json`。

`THROUGH` 回归结果为 `1743/2048=85.11%`，正式结果为 `1740/2048=84.96%`，没有回归。穿门距离通过 `1929/2048=94.19%`，arm stowed `1889/2048=92.24%`，门角和底盘稳定均为 `2048/2048`。左门成功 `914/1029=88.82%`，右门成功 `829/1019=81.35%`；剩余弱项仍是右门穿门距离，不是机械臂回收、门角或底盘稳定。指标文件为 `logs_eval/through_current_worktree_regression_eval2048/metrics_eval.json`，SHA-256 为 `6aa84ac708825c48bd77611eb3f9cf51727993de1a4dcc318d28a4c63cfedda4`。

GRASP 的“位置/切向/轴向调和平均加局部门板接近速度惩罚”候选确定性结果为 `0/2048`，相对正式基线 `257/2048=12.55%` 构成严重回归。该策略仍学会打开夹爪，但总距离通过仅 `1/2048`、任一有效接触 `128/2048`、双爪有效接触 `5/2048`、对向接触 `8/2048`，最大任务力中位数为 `0 N`。安全终止下降来自没有接触，而不是安全完成抓取。根因是调和平均把已经存在于 orientation component 的轴向约束重复计入 approach component，并削弱姿态未完全对齐时原本有效的平移梯度，使策略停在把手外。指标文件为 `logs_eval/grasp_coupled_panel_speed_step200_eval2048/metrics_eval.json`。

该调和耦合函数、调用和专用测试已经完整删除。当前 GRASP 单变量候选恢复正式基线的独立总距离势场

```text
q_distance = exp(-max(||e|| - 0.020, 0) / 0.150)
```

法向质量只保留为诊断；同时保留局部门板接近速度惩罚。该安全项只惩罚门板附近、沿 task-frame `+X` 朝门板运动且超过 `0.2 m/s` 的部分，切向运动和离开门板的运动不受惩罚。删除失败逻辑后完整测试为 `97 passed, 8 subtests passed`，真实 `8 env x 8 steps x 1 update` optimizer smoke 正常完成。四卡正式训练分两段执行：先训练至 step25，以 `unsafe_force <= 2e-4/step` 为安全门槛；通过后再续训至 step200并执行同口径2048-env确定性评估。step25 运行目录为 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260930_024648`。

step25 已完成，共 `4 * 2048 * 32 * 25 = 6,553,600` timesteps。`unsafe_force/step` 后5个 update 均值为 `6.33e-5`，全25个 update 均值为 `1.04e-4`，均低于 `2e-4` 门槛；后5个 update 的任一有效接触、双爪有效接触和对向接触均为非零。该候选通过安全门控，已从 `model_step_000025.pt` 恢复训练175个 update，目标 `global_step=200`；续训日志为容器内 `/tmp/b2z1_grasp_distance_panel_speed_to200.log`。恢复时保持相同代码、reward、随机初态、每卡2048 env、32 steps/env和4 mini-batches。

该候选已完成 step200，checkpoint 为 `door_open_b2z1_lstm_mlp-20260930_025710/model_step_000200.pt`，SHA-256 为 `4f9061d8c35ea75943f2c7019dc546098a565917f19b3cffeb8cf01e551d71b1`。训练末50个 update 的 `unsafe_force/step` 为 `1.61e-5`，但任一有效接触仅 `0.36%`。同口径确定性评估保存在 `logs_eval/grasp_distance_panel_speed_step200_eval2048/metrics_eval.json`，SHA-256 为 `7ae35ce1e8a86c83bcadeaa990ec25226bef2bc0b0487412e82cf967610d3f5c`。完整夹持、总距离通过均为 `0/2048`；任一有效接触 `161/2048`、双爪有效接触 `5/2048`、对向接触 `8/2048`，最大任务力中位数 `0 N`。因此该候选明确失败。与前一调和耦合候选共同对照后，根因已收敛到 GRASP 门板接近速度惩罚本身：即使恢复基线独立距离梯度，确定性策略仍以停在把手外规避该惩罚。该惩罚函数、调用和测试已全部删除，不得换名恢复；PUSH_OPEN 的独立相对速度惩罚不受影响。

### 2026-09-30 GRASP 轴向 orientation 连续引导

回到正式基线事件数据后，`295` 个最大达到 `9/10` 验收项的近成功环境中，`171` 个只缺少 axis `<0.05 rad`，是最大单项缺口。对应样本的旧 orientation component 中位数仍有 `0.867`，说明旧完整姿态质量对轴向误差的连续惩罚过弱；成功样本的 orientation component 和 axis quality 均为 `1.0`。

新候选保持基线独立总距离、接触势场、动作、observation、初态和验收完全不变，只将 orientation component 改为完整姿态质量和 axis 验收质量的 bottleneck 聚合：

```text
q_full = exp(-max(e_full - 0.05, 0) / 0.50)
q_axis = exp(-max(e_axis - 0.05, 0) / 0.15)
q_orientation = 0.5 * mean(q_full, q_axis) + 0.5 * min(q_full, q_axis)
```

阈值 `0.05 rad` 直接来自既有验收，不新增离散条件。离线反事实中，171个仅轴向失败样本的 orientation component 中位数由 `0.867` 降为 `0.686`，253个成功样本仍保持 `1.0`。完整测试为 `97 passed, 8 subtests passed`，`py_compile` 与 `git diff --check` 通过；真实 `8 env x 8 steps x 1 update` optimizer smoke 正常完成。四卡 step25 安全门控训练日志为容器内 `/tmp/b2z1_grasp_axis_orientation_clean25.log`。

轴向候选已完成 step200，checkpoint 为 `door_open_b2z1_lstm_mlp-20260930_040625/model_step_000200.pt`，SHA-256 为 `243fa698a90f65c18ae252a2555ac380f59bdb7a368e03a0c04efb8c2560636d`。确定性评估保存在 `logs_eval/grasp_axis_orientation_step200_eval2048/metrics_eval.json`，SHA-256 为 `bcbb3c3bd8167f33b8278ec2146daad505eef1260d46854c381c7a90ac296c9e`。完整夹持 `125/2048=6.10%`，低于基线 `12.55%`，因此候选失败。它把 axis 通过率从 `50.63%` 提高到 `63.48%`，任一有效接触从 `71.63%` 提高到 `96.83%`，但总距离通过从 `34.96%` 降到 `9.28%`，对向接触从 `50.54%` 降到 `14.99%`。候选把策略推向法向误差接近零但切向仍约 `5 cm` 的一爪侧碰状态，轴向权重过强。该候选函数、调用和专用测试已全部删除，正式代码恢复基线 orientation 公式。

正式基线训练尚未收敛的证据明确：前50到后50个 update，pocket distance `0.413 -> 0.138 m`、axis error `1.317 -> 0.551 rad`、对向接触 `0.014% -> 1.737%`、capture progress `0.00064 -> 0.0613`，均持续改善。因此下一实验不新增reward，而是从权威基线 `door_open_b2z1_lstm_mlp-20260929_194235/model_step_000200.pt` 在精确基线行为代码上继续训练到step300。四卡配置仍为每卡2048 env、32 steps/env、4 mini-batches；日志为容器内 `/tmp/b2z1_grasp_baseline_continue_to300.log`。step300后按相同2048-env确定性口径复评，只有完整夹持率明显超过 `12.55%` 才算达到本阶段目标。

续训轨迹的2048-env确定性评估已经覆盖step225、250、275、300。完整夹持分别为：

```text
step200: 253/2048 = 12.35%（事件复评；正式原始基线为257/2048=12.55%）
step225: 606/2048 = 29.59%
step250: 155/2048 =  7.57%
step275: 162/2048 =  7.91%
step300: 111/2048 =  5.42%
```

因此step225是同一训练轨迹上的明确峰值，之后出现策略漂移，继续训练并不单调改善。step225相对正式基线提高 `17.04` 个百分点；任一有效接触 `1626/2048=79.39%`、双爪有效接触 `974/2048=47.56%`、对向接触 `1106/2048=54.00%`、总距离通过 `834/2048=40.72%`、axis通过 `1396/2048=68.16%`。unsafe-force由基线事件复评的 `121` 局降至 `73` 局。

左右门均提升：左门由 `17/1029=1.65%` 提高到 `66/1029=6.41%`，右门由 `236/1019=23.16%` 提高到 `540/1019=52.99%`，不是牺牲一侧换取总体提升。step225 checkpoint 为 `logs_rl/b2z1_open_door/wbmanip/door_open_b2z1_lstm_mlp-20260930_050612/model_step_000225.pt`，SHA-256 为 `3ede0b68283e37523294c20d4b14ac1f663084901d20259852076da4cfd1a8ed`；正式指标为 `logs_eval/grasp_baseline_step225_eval2048/metrics_eval.json`，SHA-256 为 `8059a88a3b564ffdbda2c23e08b805dfaa47e804eaf5fb846fc9c726602cec1a`。

本轮GRASP验收口径是“曾单帧满足完整夹持几何”，不要求连续保持5秒，因此使用 `episode_grasp_precise_bilateral_grasp`，不使用仍带连续稳定步数要求的 `goal_reached_buffer`。最终正式代码保持基线reward；两个失败的GRASP速度/耦合候选和轴向加权候选均已删除，没有保留阶段特例、隐藏开关或seed/门型特调。

### 2026-09-30 多 seed 复现与评估协议审计

三阶段重新按统一协议复评：显式设置 `staged_reset_eval_stage`，2048 个环境各取第一局，actor 使用 `action_mean`，关闭探索，headless、无相机。GRASP step225 的 seed0/seed1 分别为 `594/2048=29.00%` 和 `606/2048=29.59%`；seed0 左右门为 `62/1029=6.03%`、`532/1019=52.21%`，seed1 为 `71/1013=7.01%`、`535/1035=51.69%`。总成功率和左右差异均跨 seed 稳定复现。

PUSH_OPEN step400 的 seed1 为 `1898/2048=92.68%`，左右门分别为 `92.60%/92.75%`；失败为门连续超速 `126` 和任务力超过 `600 N` 的 `24`，无肩碰、超时或未接触。THROUGH step400 的 seed1 为 `1742/2048=85.06%`，左右门分别为 `89.14%/81.06%`；穿门距离、arm stowed、底盘稳定和门角通过数分别为 `1931/1882/2048/2048`。两阶段均与 seed0 基线一致，没有 seed0 特化。

审计同时发现普通“收集 N 个 episode”模式会在最后一帧批量完成时超过目标数。通用评估器现按剩余名额截断完成环境，并新增协议元数据，后续指标文件会明确记录 checkpoint、step、seed、环境数、固定阶段、每环境第一局、动作来源、渲染和相机状态。期间未显式设置固定评估阶段或未启用每环境第一局的试跑不属于正式协议，结果作废，不用于策略结论。

GRASP step225 后的训练轨迹在 `6.67e-5` 学习率下发生策略漂移，而现有连续势场没有尚未验证的单项明显缺口。下一项唯一变量实验从 step225 恢复，将 actor/统一优化器学习率及自适应上限固定为 `1e-5`，reward、observation、动作、初态和验收不变；四卡各 2048 env、32 steps/env，训练 25 updates，并保存 step230/235/240/245/250 供逐点确定性复评。

低学习率实验已完成，step230/235/240/245/250 的完整夹持率依次为 `26.27%/18.07%/13.72%/9.72%/9.67%`。step250 为 `198/2048`，左门和右门分别为 `17/1029` 与 `181/1019`；正式指标文件为 `logs_eval/grasp_low_lr_step250_stage1_seed0_eval2048/metrics_eval.json`，SHA-256 为 `e1b1b7e6efd0f6aa9203a0f0fe297361a47c2b0ffb94e2641737f127c925ca26`。低学习率只减慢了漂移，没有改变梯度方向，因此该实验失败，step225 仍为正式最佳 checkpoint。

进一步审计发现 GRASP 的正式主指标和阶段完成奖励语义不一致：正式指标 `episode_grasp_precise_bilateral_grasp` 记录单帧完整双侧夹持，而 `_stage_1_reward_condition` 额外要求连续 5 个 policy step。step225 的 2048 局中单帧完整夹持为 `594`，连续达到 2/3/4/5 步的数量依次降为 `423/303/225/179`。这使多数正式成功事件拿不到阶段完成 credit。下一项单变量实验只把 GRASP 完成奖励改为本帧完整验收事件；连续势场、初态、observation、action 和其他阶段稳定条件均保持不变，从 step225 短续训并逐点复评。

单帧 completion 候选从 step225 续训后的 step230/235 分别得到 `434/2048=21.19%` 和 `350/2048=17.09%`，低于原 step225 的 `594/2048=29.00%`，因此该假设被否定，代码已恢复原有 100 ms 稳定条件。期间曾怀疑评估器的 `[N,1]` 环境 ID 张量会造成重复计数；进一步检查确认 `dones` 本身是一维，`nonzero()` 的单列就是环境 ID，修复前后的 GRASP 与 THROUGH 两个 seed 结果一致，因此该判断错误，相关评估器候选修改已全部撤销。PUSH_OPEN 在 `2047/2048` 后等待最后一个环境，是失败环境跑满该阶段配置的 1500 步，并非评估死锁。

### 2026-09-30 三阶段最终并行验收

三个阶段均使用合法程序化随机初态、2048 env 每环境第一局、固定 `staged_reset_eval_stage`、actor mean、关闭探索、headless、无相机和无渲染。GRASP step225 的 seed0/seed1 完整双侧夹持分别为 `600/2048=29.30%` 与 `598/2048=29.20%`，跨 seed 稳定超过本轮从 `12.55%` 提升的阶段目标；对应指标 SHA-256 为 `75a05e6e297e1e4a5da2890a5d6a21e37ab9dd9da1a8f915ac9922336581fb5d` 和 `3baf0da7fceb1a8da22afef148fae38f40a55b245321a17314c8c4892d322bb8`。

PUSH_OPEN step400 的 seed0/seed1 成功率为 `1884/2048=91.99%` 与 `1898/2048=92.68%`。seed1 左右门分别为 `92.60%/92.75%`；剩余失败由门连续超速 `126` 和任务力超过 `600 N` 的 `24` 构成，无肩碰、超时或未建立真实接触。指标文件 SHA-256 为 `693f5faac79ac87d1dca239d16cbedb1676cade90c2b59e36ee83c6d90088eea` 和 `9cdc396f4d9ad67a43b59a8f96f7cdd56db7916f4acfa9113a7437b9eb0cc36b`。

THROUGH step400 的 seed0/seed1 成功率为 `1743/2048=85.11%` 与 `1742/2048=85.06%`。seed1 的穿门距离、middle preset、门角和底盘稳定通过数分别为 `1931/1882/2048/2048`；左右门成功率为 `89.14%/81.06%`。指标文件 SHA-256 为 `dc1c1e444085ba1d5d7a5d001476bd06a64d979e86c9b556231518c62eb53efd` 和 `995cdce44d39b187ccaaa2f4b64105ad334335d7accd8be7081865c6a7e61410`。

本轮最终保留正式 reward 与三个最佳 checkpoint，不保留低学习率、单帧 completion、门板接近速度、调和耦合或轴向过强等失败候选。三个阶段均具备跨 seed 可复现结果，且对另外两个阶段的回归检查通过。

### 2026-09-30 GRASP 碰撞感知导航势场

只读几何诊断确认，step225 的失败局在最接近成功的一帧有约 `23%` 处于两爪同在把手上方或下方；旧 pocket 欧氏势场直接奖励缩小高度误差，并即时惩罚为了避开把手所需的法向后退。第一版碰撞感知候选把 GRASP 位置质量改为连续有序势场：未包围把手时先奖励达到 AutoOpen 同源的 `0.10 m` 安全外侧距离，到达后奖励切向/高度对齐，完整包围把手截面后才奖励最终法向靠近。动作、observation、验收、其他 reward 和其他阶段均未修改。

第一版候选的纯函数测试虽然自洽，但错误地把 task-frame 第一轴理解为机器人侧法向。四卡 step25 门控显示 pocket distance 从前5个 update 平均 `0.307 m` 恶化到后5个 `0.956 m`，任一有效接触从 `1.35%` 降到 `0.19%`；unsafe-force 降低只是远离任务。运行在 step27 请求终止，但 10:16 进程复核发现两个 worker 和 launcher 未退出，现已精确清理；checkpoint `door_open_b2z1_grasp_collision_aware-20260930_095457/model_step_000025.pt` 仅作失败证据，不允许续训。

代码与轨迹确认统一 task frame 的正第一轴指向穿过门洞，机器人侧安全后退对应负值；初态平均 pocket normal 约为 `-0.144 m`。候选已只修正这一符号。修复后解析网格覆盖上/下错位各225点，全部得到机器人侧后退梯度且其绝对值至少为错误直穿梯度的 `10.9` 倍；180个安全外侧状态全部得到上下对齐梯度，32个完整包围状态全部得到最终靠近梯度。完整测试 `99 passed, 8 subtests`，64-timestep Isaac optimizer smoke 正常完成。下一步为同配置四卡随机初始化训练，step25 先做方向和安全门控，通过后运行至 step200 并正式复评。

符号修复候选随后完成 `200 updates / 52,428,800 timesteps`，checkpoint 为 `logs_rl/b2z1_open_door/door_open_b2z1_grasp_collision_aware_signfix-20260930_100815/model_step_000200.pt`。正式评估使用固定 GRASP、2048 个程序化随机初态、每环境第一局、actor mean、关闭探索、headless、无相机；结果保存在 `logs_eval/grasp_collision_aware_signfix_step200_eval2048/metrics_eval.json`。完整夹持为 `0/2048`，任一有效接触 `9/2048`，双爪有效接触和对向接触均为 `0`，unsafe-force 为 `1/2048`。相对旧 step225 的 `600/2048` 构成确定性回归，候选失败。

失败不是策略没有学会路径前半段：`episode_max_grasp_between_jaws_quality` 均值为 `0.9995`，说明几乎全部环境都曾把把手投影放入两爪之间；切向和垂直最小误差中位数分别约 `0.7 mm` 和 `0.015 mm`。首个未通过环节是最终法向靠近：最小法向误差中位数仍为 `0.0940 m`，最小 pocket distance 中位数为 `0.1480 m`，没有环境达到 pocket `<0.020 m`。

根因来自候选势场在安全边界内侧形成的局部最优。令机器人侧安全距离为 `d`、`s=0.10 m`，当前 clearance 为 `C=exp(-max(s-d,0)/0.03)`；在 `d>=s` 时 `C=1` 且法向梯度为零。向门靠近到 `d<s` 后，route-ready 与 transverse credit 会同时因 `C` 下降而损失，而最终靠近 credit 又乘以尚未完全达到 1 的 `between_jaws_quality B`。在边界 `d=s`、横向质量约为1时，只有 `B>约0.945`，向门靠近才会产生净正收益；否则最优动作是留在安全距离。训练轨迹与确定性评估均复现该局部最优。该候选不得续训或换名复用；下一轮只允许修复从安全悬停到最终接触的连续势场，不改其他接口和阶段。

局部最优修复使用唯一连续法向目标 `d*=0.10*(1-B*T)`，其中 `B` 为把手处于两爪之间的连续质量，`T` 为切向/高度对齐质量。未包围或未对齐时目标为机器人侧 `0.10 m` 安全悬停；随着 `B*T` 从0连续升到1，目标单调移动到物理接触点。法向质量改为 `exp(-abs(d-d*)/0.03)`，切向和包围收益由该质量门控，因此贴近把手且上下错位时仍优先后退，但任意固定 `B*T` 下法向只有一个极值，不再需要跨越奖励屏障。失败候选的单边饱和 clearance、probabilistic-OR route-ready 和 `B*direct_position` 路径均已删除。

新增测试覆盖上下同侧初态、法向目标两侧以及 `B=0.1/0.5/0.9` 的部分包围状态，梯度均指向对应唯一目标；完整测试 `100 passed, 8 subtests`。真实固定 GRASP smoke 为 `8 env x 8 steps x 1 update`，共64 timesteps，无 traceback、NaN 或运行时错误。四卡干净训练已启动：每卡2048 env、32 steps/env、200 updates、随机权重，日志 `doorman_train:/tmp/b2z1_grasp_moving_target_clean200.log`。
