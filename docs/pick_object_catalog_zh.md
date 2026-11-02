# 待抓取物体图册

[当前PDF：物体位图、中文矢量文字，约5.7 MB](../artifacts/pick_object_catalog/pick_object_catalog_hybrid_zh.pdf)

当前按外部/背筐统一6 mm间隙更新，共137个资产、59个类别、12页；每页4行3列，末页5个、其余留空。每个物体包含正视、右视、俯视和三维等轴斜视图，A3原尺寸全册统一比例1:6.5。包括78个Lightwheel资产（含KitchenRoom调味罐和瓶盖2个）和11个Extwin资产；不包括当前未通过的4个实例。准入规则和完整复核证据见[统一抓取间隙](grasp_clearance_zh.md)。

排序先按原始来源：YCB、GSO、光轮Lightwheel/RoboCasa小物体包、光轮Lightwheel/KitchenRoom厨房包、Extwin。每个来源内按中文类别归组、同类别按原实例编号排序。每个物体标注中文类别、原始来源、原始尺寸和中文视角名，保留英文资产编号供回查；不把不同来源的同类别混排。

物体视图保留实际模型的纹理，以位图嵌入；名称、尺寸、视角标签和比例尺均为原生矢量文字与线条，可选择、复制。位图区域不包含截图文字。视角沿资产原始坐标系，不代表语义上的正面。

图册展示当前启用且通过形状筛选的基础资产，不枚举连续随机缩放、调色，不表示每个资产都已通过物理抓取验证。

## 旧版输出

- [旧48资产总览PNG，6540×9978](../artifacts/pick_object_catalog/assets/pick_object_catalog/all_objects_four_views_same_scale.png)
- [旧48资产HTML总览](../artifacts/pick_object_catalog/pick_object_catalog_zh.html)

图片、PDF、HTML及生成清单均存于Git已忽略的`artifacts/pick_object_catalog/`，不进入版本库。本文件仅维护轻量说明和链接；克隆仓库后需要在本地重新生成或另行获取二进制文件。
