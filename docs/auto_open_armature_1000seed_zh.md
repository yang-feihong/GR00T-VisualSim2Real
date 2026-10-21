# AutoOpen 腿部惯量修复后的 1000-Seed 验证

## 配置

- 状态：已完成，1000 条全部落盘并完成严格验收。
- 日期：2026-10-21，Asia/Shanghai。
- Seed：1700–2699，共 1000 个。
- 固定入口：`gr00t/rl/scripts/run_auto_open.sh`。
- 模式：`canonical`，推门完整流程至 `THROUGH`；无相机，不启动补图或正式发布。
- 并行：4 张 GPU，每卡 1 个进程、250 个环境。
- 当前修复：仅为腿部 actuator 增加 `armature=0.1`，没有迁入指令门控，没有同步其他 Gym 动力学参数。
- 测试快照边界：本批完成于三个 preset 更新之前，测试时 middle 为 `[-1.57, 1.6, -0.63, -1.1, 0, 0]`；之后更新为 `[-1.57, 1.75, -0.785, -0.93, 0, 0]` 的版本尚未重测 1000 seeds，不能直接沿用本批通过率。
- 成功定义：完整严格数据验收通过；仅控制流程完成不算成功。图像不纳入本轮审核。
- 输出目录：`datasets/auto_open/canonical/auto_open_v1_seeds1700-2699_canonical_no_camera_20261021T134745`。

## 启动命令

```bash
SEED_START=1700 SEED_END=2699 AUTO_OPEN_MODE=canonical \
DOOR_MOTION=push STOP_AFTER_STAGE=THROUGH GPU_IDS=0,1,2,3 \
bash gr00t/rl/scripts/run_auto_open.sh
```

## 结果

依据批次 `acceptance_report.json`：

| 指标 | 结果 |
| --- | ---: |
| 请求 / 完成 / 唯一 seed 数 | 1000 / 1000 / 1000 |
| 严格数据验收通过 | **887/1000（88.7%）** |
| 严格数据验收失败并隔离 | 113/1000（11.3%） |
| 采集墙钟耗时 | **2492 s（41 min 32 s）** |
| 验收 exceptions | 0 |
| 非足部机器人触地 episode 数 | **0** |
| APPROACH / ALIGN_PANEL 流程失败 | **0 / 0** |

耗时采用固定入口生成的 `collection_wall_time_s`，包含四卡启动、仿真、采集和 shard 落盘，不是纯物理推进时间；本轮没有补图。
相对 `auto_open_gpu_batch_status_zh.md` 记载的正式基线 696/1000、2676 s，通过率增加 19.1 个百分点，耗时减少 184 s（3 min 4 s）。
相对紧邻本次修复前的统一站位控制对照 712/1000、2750 s，通过率增加 17.5 个百分点，耗时减少 258 s（4 min 18 s）。后者原记录在 `.git/autoopen_unified_station_checkout/docs/auto_open_station_control_ab_zh.md`，不是正式发布数据的新基线。
没有将流程完成但数据验收不通过的样本计入成功。

## 失败分布

按每条失败 episode 的首个阻塞原因唯一归类，合计 113。该表不是可重叠的全部缺陷计数，也不把终止段缺失这类失败后的伴随结果单独再算一次。

| 首个阻塞原因 | episode 数 |
| --- | ---: |
| 异常环境 / 阶段外接触 | 48 |
| PUSH_OPEN 流程失败 | 24 |
| GRASP_HANDLE 流程失败 | 8 |
| 底盘指令短时往返 | 8 |
| CONTACT_PANEL 流程失败 | 8 |
| 机械臂目标短时往返 | 6 |
| 握把阶段底盘移动 | 4 |
| BASE_ADVANCE 流程失败 | 3 |
| CENTER_HANDLE 流程失败 | 3 |
| PUSH_PULL_DOOR 流程失败 | 1 |
| 合计 | **113** |

流程失败的完整诊断计数如下，部分同时触发更早排列的数据审计缺陷，因此与上表不一一等数，也不能与上表相加：

| 阶段 | 具体原因 | episode 数 |
| --- | --- | ---: |
| PUSH_OPEN | 连续 300 帧门无进展 | 17 |
| PUSH_OPEN | 手指接触力超过 600 N | 10 |
| PUSH_OPEN | 连续 3 帧门速超限 | 8 |
| PUSH_OPEN | 验收超时 | 2 |
| CONTACT_PANEL | 验收超时 | 10 |
| CONTACT_PANEL | 接触目标逆映射失败 | 3 |
| GRASP_HANDLE | 验收超时 | 8 |
| BASE_ADVANCE | 验收超时 | 4 |
| CENTER_HANDLE | 验收超时 | 3 |
| PUSH_PULL_DOOR | 验收超时 | 1 |

另外，全量可重叠审计缺陷中，机械臂目标短时往返为 10 条、穿透超限为 1 条；其余可能已经因其他原因被隔离，不能只看首因表而误称这些问题不存在。
本轮只重测并统计，没有继续修改这些残留控制或验收问题。

## 验证产物

- `acceptance_report.json`：1000 条逐 episode 严格验收与失败原因。
- `manifest.json`：完整批次清单。
- `workflow_source.json`：本轮工作区源码指纹。
- `logs/rank*-control/seed_*/metadata.json`：FSM 失败阶段、失败原因与验收诊断。
- `shards/episode-seed*`：全部通过与隔离样本，图像不在本轮采集范围。

已核对 seed 集合恰为 1700–2699，无缺失或重复。四个进程均结束，无 GPU 计算进程残留。
