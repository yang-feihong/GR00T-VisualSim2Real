# Lightwheel 抓取资产扩展

## 当前状态

用户已确认仅用于非商业研究。完成外部/背筐统一6 mm及全部历史采样尺寸复核后，Lightwheel现有78个启用资产、15类：小物体包76个，KitchenRoom调味罐和瓶盖2个；其中66个有背筐支撑形态。加上48个YCB/GSO资产与11个Extwin资产，默认库共137个资产、59类。这里的准入不等于逐资产闭环物理抓取验证。当前规则与恢复证据见[统一抓取间隙](grasp_clearance_zh.md)。

| 来源 | 实际文件 | 完整性 | 当前检查 |
| --- | --- | --- | --- |
| RoboCasa官方分发的Lightwheel小物体包 | 791939656字节 | SHA256 `a1db739b0e80fc4d56d4be27d89dba3aa60000e9648c5607874700e0f14b7955` 已核对 | 968个模型中210个通过粗筛；105个试导入，当前统一6 mm后76个启用 |
| Lightwheel KitchenRoom | 699682909字节 | SHA256 `721fb9e47d09b2c43a7db396c1d63c659e987d5925db0a5e78cf4fd40cd0b0b0` 与官方Git LFS一致 | 已检查6个独立模型，3个进入试导入/准入，当前2个启用 |

## KitchenRoom 初筛

| 独立资产 | X×Y×Z，毫米 | 当前结论 |
| --- | --- | --- |
| SM_P_Flavour_02 | 70.89×70.89×190.93 | 含玻璃瓶身和不同部件材质，当前不转为单一不透明材质，明确拒收 |
| SM_P_Flavour_04 | 55.48×55.48×93.43 | 原始分组及多边形已正确处理；保留三维顶点及单位，通过GPU准入，已启用 |
| SM_P_Flavour_03_Cover | 36.64×36.64×17.47 | 独立瓶盖，补充静态稳定摆姿后通过6 mm检查，已恢复 |
| SM_P_Flavour_03 | 107.13×97.93×305.64 | 超过当前最大尺寸，不缩小救回 |
| SM_P_Jar_01 | 93.29×93.29×112.17 | 初筛通过，但横截面超过当前单凸包资产转换要求，且为空心玻璃罐，不能直接当实心小物体发布 |
| SM_P_Tray_01 | 215.61×325.84×9.12 | 超尺寸且为薄片，不纳入 |

## 边界

- 使用`lightwheel_`前缀，保留来源、归档哈希、CC BY-NC 4.0许可及非商业研究范围，不将镜像站的软件许可误报为资产商用许可。
- 不改推拉门或抓取控制；通过既有`asset.json`、USD、变体和GPU准入进入相同采集链路。
- 单网格导入器对透明、多部件或未烘焙变换显式拒收，不默默丢零件、变材质或改变尺寸。
- 下载、渲染、PDF和逐资产报告保留于Git忽略的`artifacts/lightwheel_acquisition/`及资产数据目录，不提交二进制。

## 已启用类别

| 类别 | 实例数 |
| --- | ---: |
| 鸡腿 | 8 |
| 曲奇面团球 | 10 |
| 奶油奶酪棒 | 1 |
| 热狗面包 | 10 |
| 冰块 | 1 |
| 柠檬瓣 | 1 |
| 棉花糖 | 5 |
| 芥末酱瓶 | 7 |
| 胡椒瓶 | 7 |
| 盐瓶 | 7 |
| 虾 | 1 |
| 调味瓶/罐（KitchenRoom） | 1 |

保留原始尺寸；原尺寸存在支撑面夹取候选后，再分别检查0.95和1.05倍等比变体，只保留通过的变体。透明、空心容器及当前不支持的多部件材质没有强行变成实心单物体。

## 验证与交付

- 首次导入日志保留原106个候选的结果；它不再代表当前准入数量，当前以[统一6 mm结果](../artifacts/unified_grasp_clearance/summary.json)为准。
- 共享入口、资产库及USD相关测试58 passed；最新转换/纹理/真实资产核对专项19 passed（6.33秒）。实际资产USD、碰撞缓存、默认采样池、变体及背筐过滤均已检查；新增资产的凸包顶点均不超过10000。
- [首次导入原始结果](../artifacts/lightwheel_acquisition/admission_results.json)、[当前统一6 mm统计](../artifacts/unified_grasp_clearance/summary.json)。
- [最新四视图PDF](../artifacts/pick_object_catalog/pick_object_catalog_hybrid_zh.pdf)：137个资产、12页、约5.7 MB；每页4行3列，末页5个，其余留空；按来源/类别排序，中文矢量说明，统一比例，物体为位图。
- [Extwin扩展结果](extwin_pick_assets_zh.md)：已下载24个样本，当前统一6 mm下11个启用。

## 颜色纹理修复

用户指出PDF中的光轮资产呈统一灰色后，确认导入错误：缺少有效MTL贴图的OBJ被`trimesh`配上默认2×2灰图，转换器误将该占位图视为有效材质，因而没有使用MJCF中明确指定的原始贴图。错误发生在资产转换，不只影响PDF显示。

已修改转换器：显式提供的源贴图优先；若明确有贴图却没有UV则报错，禁止静默丢弃纹理。已恢复92个已导入且有UV的Lightwheel资产的原贴图（包含全部80个启用资产），逐像素核对来源，逐项核对源UV，且USD几何、碰撞缓存、准入元数据文件哈希完全未变。另有1个原已禁用资产无UV，继续禁用。

新增占位贴图覆盖与缺UV拒收回归测试；纹理检查及资产库相关测试33 passed，16.54秒。[逐资产核对结果](../artifacts/lightwheel_acquisition/texture_repair.json)。PDF已按修复后的实际资产重新渲染，替换同路径旧版本；无需重新进行几何准入。

## 来源

- [Lightwheel开源资产与署名要求](https://github.com/LightwheelAI/Lightwheel-simready-asset)
- [Lightwheel KitchenRoom及许可](https://github.com/LightwheelAI/Lightwheel_Kitchen)
- [RoboCasa官方分发清单，固定提交](https://github.com/robocasa/robocasa/blob/be22d659b02db8f6d7f3a3c3edc742934fdcbaae/robocasa/models/assets/box_links/box_links_assets.json)
