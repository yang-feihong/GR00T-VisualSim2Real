# VLNVerse 场景接入

## 当前能力

`/mnt/data2/yangfeihong/vlnverse_scenes` 可以直接接入当前 Isaac Sim 运行时，无需转换资产：

- 场景主文件为米制、Z-up USD，内部 Mesh 和材质使用相对引用；
- 主 USD 已包含静态 triangle-mesh collision；
- 共发现 262 个主场景，其中 176 个同时包含 `occupancy.png`、`occupancy.json` 和 `rooms.json`；
- 默认只从这 176 个场景中采样，以保证机器人初态可以落在已知自由空间；
- 176/176 个场景均通过 B2Z1 半径 0.55 m 净空出生点检查。

VLNVerse 作为静态背景场景加载。机器人、相机和未来的动态抓放物体仍由任务框架创建，不把整套房间注册成一个刚体，也不与 AutoOpen 逻辑耦合。

每个源 USD 自带一个 `physicsScene` prim。加载器会停用这些内嵌 PhysicsScene，只保留碰撞 API，由框架创建的唯一 PhysicsScene 统一步进；否则多 env 时会触发 Isaac Sim 不支持的 per-scene stepping。

## 容器挂载

仓库外层的 `run.sh` 默认将：

```text
/mnt/data2/yangfeihong/vlnverse_scenes
```

只读挂载到：

```text
/datasets/vlnverse_scenes
```

可通过 `VLNVERSE_SCENES_DIR` 和 `VLNVERSE_MOUNT_POINT` 覆盖两端路径。挂载源被写入容器 label；路径变化或旧容器缺少该挂载时，`run.sh` 会要求重建对应容器配置，不复制 292 GB 资产。

## 配置入口

任意继承 `config/base.yaml` 和 `env/base_task.yaml` 的 Isaac Sim 任务，只需增加：

```text
static_scene=vlnverse
```

例如：

```bash
python -u gr00t/rl/scripts/debug_b2z1_fixed_command.py \
  +exp=b2z1_flat_lowlevel \
  static_scene=vlnverse \
  seed=1700 \
  num_envs=2 \
  headless=True \
  +num_debug_steps=100 \
  '+velocity_command=[0.0,0.0,0.0]'
```

默认是按 seed 随机分配场景。完整配置位于：

```text
gr00t/rl/config/static_scene/vlnverse.yaml
```

固定单个场景：

```text
static_scene.selection.mode=fixed
static_scene.selection.fixed_scene_id=kujiale_0005
```

限制白名单：

```text
'static_scene.selection.scene_ids=[kujiale_0005,kujiale_0006]'
```

按排序循环分配：

```text
static_scene.selection.mode=cycle
```

## 随机化语义

场景分配在 SimulationApp 创建时完成：

- 同一个 `seed + env_id` 始终选择同一场景；
- random 模式按无放回乱序循环，先覆盖更多不同场景；
- 不同场景需要 `replicate_physics=false`，运行时会自动设置；
- 最大户型约为 22.6 x 20.2 m，运行时会把 env 间距至少提升到 30 m。

机器人初态在每次 reset 时从该 env 对应 occupancy 的安全候选中选择：

- 白色像素表示自由空间；
- 默认保守要求采样点四周 0.55 m 的方形足迹范围均为自由空间；
- 像素坐标根据 `occupancy.json` 的 `scale/lower/upper` 转为场景局部米制坐标；
- x、y 和 yaw 均随机化，z 继续使用机器人任务自身的标准初始高度；
- 候选在启动时一次性计算，reset 热路径只做 GPU 张量索引。

运行时通过 `simulator.static_scene_metadata` 暴露每个 env 的 scene ID、USD 路径、seed 和出生净空参数，通过 `simulator.static_scene_last_spawn_poses` 暴露最近一次 `(x,y,yaw)` 初态，便于未来数据 provenance 直接记录。

## 抓放任务边界

当前链路只负责室内静态场景和机器人安全初态，不擅自定义抓放专家流程。后续类似 AutoOpen 的抓放流程应在此基础上增加：

1. 从房间、support surface 或任务配置选择工作区域；
2. 将待抓物体作为独立动态刚体生成，不能直接把场景内静态装饰物切成动态物体；
3. 基于目标物体生成 approach、grasp、lift、place 等同层 FSM 阶段；
4. 把 scene ID、初态、物体资产和目标位姿写入每个 episode 的机器可读 metadata。

整套高精度碰撞场景不适合在 episode 中热替换。推荐一个进程创建一批静态场景，episode reset 只随机化机器人初态和动态物体；切换整套户型时重建 SimulationApp。

## 已完成验证

- 纯函数测试覆盖场景发现、缺失元数据过滤、固定/循环/随机选择、batch-prefix 确定性和 occupancy 净空采样；
- 真实资产扫描：176/176 个导航场景均可生成 0.55 m 净空出生点；
- Isaac Sim 2-env smoke：seed 1700 分配 `kujiale_0109`、`kujiale_0077`，实际间距 30 m；
- 两套 USD 完成 PhysX 初始化，机器人从 occupancy 初态 reset 并执行控制步；
- 固定 `kujiale_0005` 的相机 smoke 成功输出 426 x 240 RGB，Mesh、纹理和 MDL 相对引用生效；
- 停用资产内嵌 PhysicsScene 后，当前缓存下单场景无渲染 PhysX 启动约 2 s、两个不同场景约 10 s、单场景带渲染约 21 s；日志中无 per-scene stepping 错误。

带相机 smoke 输出保留在：

```text
logs_rl/vlnverse_scene_render_smoke_20261013/ego_camera.png
```
