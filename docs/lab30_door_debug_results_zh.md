# 推拉门 Debug 结果

## 本轮结论

标准推门、拉门seed1700–1703均使用正式debug入口，四卡六视角，trunk滤波开启，**流程8/8完成，严格8/8通过，无拒收项**。下方八个视频已经全部替换为本轮最终携臂版本，没有追加旧视频。

**整批验收已完成：**最终推门严格871/1000（87.1%）、2166秒（36分06秒），全部1000条逐episode验收报告与修复前完全一致；拉门严格799/1000（79.9%）、3021秒（50分21秒）。拉门比起始参考81.8%低1.9个百分点，按用户允许约80%的可比范围收口，不把79.9%写成80.0%。[总文档](auto_open_gpu_batch_status_zh.md)已更新，提交和远程换版状态见[台账](door_baseline_validation_progress_zh.md)。

| 模式 | 批次 | 流程完成 | 严格通过 | 采集耗时，不含后续视频拼接 |
| --- | --- | ---: | ---: | ---: |
| 推门 | 20261103T022754 | 4/4 | 4/4 | 1408秒，23分28秒 |
| 拉门 | 20261103T022744 | 4/4 | 4/4 | 1859秒，30分59秒 |

八条记录均无异常接触或穿透违规。推门1703本轮不再卡在二次前移，最终门角87.724°且已完整穿门。拉门1702也严格通过，不再因释放末尾连续延续的后部可分离把手接触被拒收；实际安全分类及穿透上限仍保留。

上表多个debug批次并发，不作为独占四卡canonical的性能基准。两个1000seed已按顺序、各自独占四卡执行，无图像及补图。

## 自主修复边界

起点版本拉门穿门时，底盘尚在侧移/转向就把机械臂展开为middle，导致腕部/前臂扫到门框。投影10cm条件按用户定义约束的是假想居中、垂直门框的狗身，不能以修改这个条件或豁免撞框来掩盖问题。

仅拉门THROUGH增加紧凑携臂：保留入口实际q1，q2–q6目标为`[1.45,-0.8,1.1,0,0] rad`，先折臂，再从实际机身位姿计算原穿门轨迹；机尾清空门框1m后单向恢复middle。使用已有JOINT接口和GPU张量，不增加在线碰撞搜索，不改底盘路线、10cm放行条件、接触或终末验收。

该修改先在临时区验证：同一典型1704–1707，原版严格0/4，保持原收臂姿态3/4，紧凑携臂4/4；标准推拉严格8/8后才迁回。454个运行文件与验证源码逐字节一致。推门四条全流程、拉门四条进入THROUGH前的实际状态和控制输出，均与起点记录一致（逐元素1e-6容差）；不是把早期阶段也一起改了。证据在`artifacts/door_baseline_validation/runtime_migration_equivalence.json`及`final_debug_media.json`。

大批次并非穿门零碰撞：拉门穿门首因由360降到23，其中机械臂碰门框19、腿碰门板4。19条中13条在底盘零指令折臂时发生、6条在侧移/转向时发生；原离线统一比例插值没有精确模拟JOINT接口的逐关节独立限速，检查存在缺口。替代q2姿态只有离线结果，未进入当前基线。所有这些碰撞仍严格拒收。详见总文档的剩余问题，不能用八条debug通过掩盖它们。

## 进入穿门的实测

固定89°已由门框平面投影间隔≥10cm替代。投影包含门板厚度；机器狗按假想对准门框中心、垂直门框的姿态，使用初始化狗身碰撞包络（含机身、背筐、腿部，不含机械臂）。在线判定、规划预估及数据验收使用共用GPU几何。

下表读取原始逐帧Parquet。间隔取首个THROUGH行的上一行post-state，门角取首个THROUGH行的pre-state，二者是同一个判定时刻，不拿最终门角代替入口条件。

| Seed | 推门入口门角 | 推门投影间隔 | 拉门入口门角 | 拉门投影间隔 |
| --- | ---: | ---: | ---: | ---: |
| 1700 | 78.550° | 10.344cm | 150.000° | 29.991cm |
| 1701 | 81.834° | 10.624cm | 145.351° | 21.568cm |
| 1702 | 79.686° | 10.266cm | 116.192° | 24.912cm |
| 1703 | 79.642° | 10.315cm | 139.538° | 26.167cm |

拉门进入THROUGH前仍需完成收臂，门会在收臂期间继续转动，所以穿门入口角度不等于触发收臂的角度，也不是仍按89°等待。门角和投影间隔均通过原始记录核实，未修改旧数据或放宽其他验收。

完整开门离线回归476项通过；动力学、锁舌及资产补充62通过、1跳过。迁回工作区后73项针对性回归再次通过。物理闭环结果以上述新两批为准，不以单测替代闭环。

## 六视角视频

本报告只保留本轮八个已完成视频。分辨率1440×720、50fps；推门帧数2194/2056/2319/2205，拉门3575/3419/3570/3512，与控制记录一致。逐视频抽检首/中/后段，画面非空且随时间变化。视频编码未阻塞canonical实验。

### 推门1700

全流程完成，严格通过。

<video controls preload="metadata" width="960" src="../datasets/auto_open/debug/auto_open_debug_push_v1_seeds1700-1703_episodes4_six_camera_20261103T022754/six_camera_mosaics/seed1700_six_camera_mosaic.mp4"></video>

### 推门1701

全流程完成，严格通过。

<video controls preload="metadata" width="960" src="../datasets/auto_open/debug/auto_open_debug_push_v1_seeds1700-1703_episodes4_six_camera_20261103T022754/six_camera_mosaics/seed1701_six_camera_mosaic.mp4"></video>

### 推门1702

全流程完成，严格通过。

<video controls preload="metadata" width="960" src="../datasets/auto_open/debug/auto_open_debug_push_v1_seeds1700-1703_episodes4_six_camera_20261103T022754/six_camera_mosaics/seed1702_six_camera_mosaic.mp4"></video>

### 推门1703

全流程完成，严格通过；最终门角87.724°，无需达到旧89°。

<video controls preload="metadata" width="960" src="../datasets/auto_open/debug/auto_open_debug_push_v1_seeds1700-1703_episodes4_six_camera_20261103T022754/six_camera_mosaics/seed1703_six_camera_mosaic.mp4"></video>

### 拉门1700

全流程完成，严格通过。

<video controls preload="metadata" width="960" src="../datasets/auto_open/debug/auto_open_debug_pull_v1_seeds1700-1703_episodes4_six_camera_20261103T022744/six_camera_mosaics/seed1700_six_camera_mosaic.mp4"></video>

### 拉门1701

全流程完成，严格通过。

<video controls preload="metadata" width="960" src="../datasets/auto_open/debug/auto_open_debug_pull_v1_seeds1700-1703_episodes4_six_camera_20261103T022744/six_camera_mosaics/seed1701_six_camera_mosaic.mp4"></video>

### 拉门1702

全流程完成，严格通过。

<video controls preload="metadata" width="960" src="../datasets/auto_open/debug/auto_open_debug_pull_v1_seeds1700-1703_episodes4_six_camera_20261103T022744/six_camera_mosaics/seed1702_six_camera_mosaic.mp4"></video>

### 拉门1703

全流程完成，严格通过。

<video controls preload="metadata" width="960" src="../datasets/auto_open/debug/auto_open_debug_pull_v1_seeds1700-1703_episodes4_six_camera_20261103T022744/six_camera_mosaics/seed1703_six_camera_mosaic.mp4"></video>
