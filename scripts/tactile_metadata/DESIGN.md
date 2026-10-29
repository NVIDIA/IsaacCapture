<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Tactile Metadata Editor 设计

Tactile Metadata Editor 是供触觉手套厂商使用的本地 GUI 工具，用于生成、修改、验证 TactileMetadata，并结合 URDF 和 Isaac Capture 数据检查手部运动与触觉分布。首版支持手套；设备来源和几何映射分别通过适配接口接入，以便跟随 schema 扩展。

本文按用户界面和功能、依赖、代码架构、模块内部实现及约定、实现代理规范组织。以下是实施规格，尚不是已实现功能。

## 1 用户界面和用户功能设计

### 1.1 输入与结果

文件输入只支持 URDF 和 TactileMetadata JSON。STL/OBJ 可以是 URDF 引用的网格资源，不作为独立设备输入；不提供 CSV 或裸二进制导入。载入 URDF 后，可新建 metadata，或把已有 metadata JSON 分配到明确的手侧并继续编辑。

典型操作是：载入 URDF 并确定手侧，在 Segments 页配置关节与绑定姿态，在指腹标记范围并输入 4×4，批量生成 16 个 taxel，设置输出顺序及最多四个关节的权重，再通过模型运动或实时数据检查。不要求厂商预先持有逐 taxel 位姿、四元数或 skinning weights。

保存、备份与导出只写触觉 metadata 数值，使用同一种 schema JSON 表示；可选 SDK 二进制仅编码同一组 metadata 数值。JSON 本身可重新打开编辑，不区分“可编辑版”和“纯数值版”。不生成编辑器工程、`.editor.json`、内部 ID 对应表或检查报告文件。

JSON 仅采用 [`tactile_metadata.fbs`](../../src/core/schema/fbs/tactile_metadata.fbs) 定义的字段、嵌套结构及 union 类型标识：顶层为 `geometry_mapping_type`、`geometry_mapping`、`segment_source_type`、`segment_source`；内容为 segment 绑定位姿、taxel 绑定位姿、四槽 segment 索引与权重、`joint_ids`。不添加 `editor` 扩展或其他工具专用字段。Python 使用 `json.load()` 读取；需要 SDK 编码时，在隔离子进程中进行现有绑定所需的数组和枚举转换，调用 `LBS`、`HandSkeletonRef`、`TactileMetadata` 构造器及 `to_bytes()` 生成符合该 schema 的 FlatBuffer。转换复用[现有绑定](../../src/core/schema/python/tactile_metadata_bindings.h)，属于常规实现工作，不构成待定产品规格。

URDF 路径、package 映射、q0、腕部校准、segment 与链接关联、手侧分配、实时数据源与关节映射、内部标识和显示状态只存在于当前会话，不写入 metadata 或另存配置文件。检查状态、草稿标记和 revision 只显示在界面或 CLI 输出中。

实时输入首版接入 TactileDataArray/TactileMetadata、HandPose 和可选 JointStateOutput。真实设备协议接入、原始 ADC/压力到牛顿的转换、设备控制及物理标定不在工具内完成。工具只接收现有 Isaac Capture 数据；编辑、保存和导出均不向 runtime 或生产设备发送 metadata、控制命令或其他数据。

### 1.2 窗口布局与显示

一个浏览器窗口包含三维画布和右侧标准 medium 宽度面板。面板顶部显示文件、已固定的 URDF 手侧及当前模式；下方是模型、Segments、布局、绑定、实时、导出六个页签。长表单滚动，复用 Viser 客户端及标准控件，不维护独立前端。

```text
┌──────────────────────────────────┬──────────────────────────────┐
│                                  │ URDF 名称；手侧：双手（只读） │
│        三维模型与手骨架           │ Metadata 文件 *        保存  │
│                                  │ 编辑对象：[左手] [右手]       │
│ taxel、选中关节、法线和切线       │ 模式：编辑 / 预览 / 实时 / 暂停│
│                                  ├──────────────────────────────┤
│ 相机：旋转 平移 缩放 重置         │ 模型 Segments 布局 绑定 实时 导出│
│ 选择：单选 / 框选 / 穿透选择      │ 当前页控件与选择属性           │
│                                  ├──────────────────────────────┤
│ Info：草稿错误状态及具体原因      │ 连接和数据状态               │
└──────────────────────────────────┴──────────────────────────────┘
```

三维场景采用右手坐标约定：+X 向左、+Y 向上、+Z 向前，长度单位为米。模型为中性色，所有 URDF 链接使用固定不透明度（opacity）0.6，高亮只改变颜色。编辑时 taxel 按主影响关节着色，无绑定为灰色，选中轮廓为黄色，错误轮廓为红色。实时触觉用力向量模长映射蓝至红，默认范围 0–30 N，可调整上限，超限饱和；无有效数据为灰色，不能显示成零力。属性面板显示 fx、fy、fz 和模长，单位 N。

球形标记大小只影响显示，不改变 sensing location。选中 taxel 显示 +Z=外向法线 N、+X=传感器切线 T、+Y=N×T；X×Y=Z。双手使用同一右手坐标约定。多关节 LBS 动画只保证位置，不把混合旋转解释为协议定义的唯一方向。

六页表单始终显示并编辑 Document 中的草稿值，包括 PREVIEW、LIVE、PAUSED 及连接中断时。三维视图的 metadata 来源切换不改变表单数据；接收值只有通过“记录 metadata”显式替换所选草稿分组后才成为表单值。力值和连接状态是单独的只读实时信息。

允许草稿包含错误。出错字段及所属分组显示错误状态和原因，三维视图下方的紧凑 Info 区汇总当前手侧、分组、实体和原因，并可定位对应字段。保留能显示的有效几何；缺失或无效的位姿不以假位置绘制。错误不阻止继续编辑或替换数据，只阻止正式导出。

### 1.3 文件、URDF 与左右手

启动页提供“载入 URDF”和“打开 metadata JSON”。载入 URDF 时选择左手、右手或双手，完成后只读；替换 URDF 或改变手侧需要创建新文档，先处理未保存内容。URDF 不携带标准手侧字段，该选择是用户对模型的明确声明。

一个会话使用一个 URDF。双手选择要求模型实际包含左右手结构，随后分别指定腕部链接和关联链接；不能把单手模型自动镜像成双手。独立左右 URDF 分别打开文档。双手文档中的左右 metadata 独立，但共享同一 URDF 和绑定关节配置；共享关节的修改会明确列出受影响手侧。

顶部“编辑对象”只选择当前要编辑的 metadata，不修改模型手侧。单手不显示切侧控件；双手可以显示当前侧或双侧，另一侧 taxel 只读。任何显示排布偏移均不参与元数据计算。两侧数量与输出排列可以不同，各自的 taxel 索引从 0 开始。

新建 metadata 不要求输入预期数量，界面从当前 taxel 记录计算 N。检查与实时数据比较时才核对数量。全工具使用“taxel 数量”“taxel 索引”和“输出顺序”，不以 channel 表示 taxel。

打开 metadata JSON 时，用户通过界面或 `--sides` 明确分配左侧或右侧，不能从关节名称推断手侧。重新打开只恢复 metadata 数值及 segment、taxel 数组行序；需要表面操作和模型动画时，在当前会话重新载入 URDF、关联链接并设置腕部参考。仅有 JSON 时仍可查看和修改数值。

模型、JSON 和 package 资源均使用本地路径，不要求逐个上传网格。未保存标记只比较 metadata 数值；仅改变模型、校准、数据源或显示配置不构成 metadata 未保存。打开新文档、替换一侧 metadata 或退出时，有未保存数值则提供保存、放弃、取消。双手分别保存两个 metadata JSON；“保存全部”处理两侧，失败不得将任一侧误标为已保存。

### 1.4 模型页

页面提供 URDF 路径、统一正比例缩放、package:// 的本地目录映射和资源状态。缩放同时作用于网格与关节平移，不改变关节角或 metadata 数值。URDF 初始映射将其 +Z、+Y、+X 分别映射到场景 +Y、+X、+Z。可检查模型树、关节类型与范围；缺失资源列出具体路径，不从网络寻找模型。

用户为每侧选择腕部链接，并拾取腕中心 O、朝前臂的参考 F、朝手背的参考 D，建立来源根坐标系。双手分别校准，界面显示两套参考轴，不按左/右自动变号。关联链接限制该侧拾取范围；共享掌臂链接可以作为参考，但另一侧表面不能成为默认拾取对象。

### 1.5 Segments 页

页面位于布局之前，包含“绑定姿态”和“Segments”两组控件。

绑定姿态组提供 URDF 自由关节滑块、数值输入和“设为绑定姿态”。fixed 不可编辑，mimic 跟随主关节，限位始终可见。此姿态 q0 是静态布局和 bind poses 的共同参考；实时角度及运动预览角度不会自动写入 q0。

Segments 组从 hand.fbs 的 HandJoint 枚举选择需要的关节，排除 NUM_JOINTS。新草稿以腕部为根，在中性张开姿态中初始化全部 26 个 OpenXR 手部关节；左右手使用镜像姿态。初始显示映射将手部 -Z、+X、+Y 分别映射到场景 +Y、-X、-Z。表格显示语义名称、输出行号、关联 URDF 链接和参考系状态。合法 metadata 每侧同名语义只出现一次，不要求完整骨架，WRIST 使用根恒等位姿；草稿违反这些要求时显示错误并保留值。选择一行会高亮关节球、坐标轴和关联链接。

新建 segment 时选择语义和链接，在模型上定位关节中心及参考方向，通过控制器调整位置与扭转。已有 metadata 的 B_j 已给出坐标：选择目标链接后计算固定局部修正，保持原 B_j，而不是把数值吸附到 URDF 原点。

允许删除仍被引用的 segment；未解决的引用留在草稿中，绑定页和 Info 区显示原因，不隐式重绑。

设置新 q0 或修改腕部参考系时，先预览模型和附件变化；现有 B_j、taxel 的绑定根位姿和权重保持数值不变，确认后重算关节局部附件并使检查过期。实际 B_j 的修改通过关节参考系控制器完成。工具不隐式重投影、蒙皮重绑定或修改生产端 metadata。偏离模型的点保留并提示用户调整。

### 1.6 布局页

页面提供“批量添加”“放置单点”和所选 taxel 的属性。没有区域名、区域列表或持久化批次；批量添加只是一次生成操作，导入 metadata 与新建结果拥有相同编辑能力。

批量添加只有一种方式：输入范围和行列数量。操作依次为：

1. 选择允许投影的链接和表面，输入 rows、cols。
2. 拾取首点中心 O、首行末点中心 C、首列末点中心 R。显示列跨度、行跨度及推导间距；可直接修改跨度数值，仍属于同一个生成模式。
3. 选择 row-major、column-major、row-snake 或 column-snake、行列反向和起始 taxel 索引，预览后应用或取消。

范围以首末 taxel 中心定义，不是传感片外边缘。单行只需 O/C，单列只需 O/R，单点只需 O 和切线方向，避免除以零。可在预览中排除实际不存在的单元；被排除单元不产生 taxel，投影失败单元则保留为有问题的候选，不能混为一谈。

预览显示位置、数量和输出索引，错误标红。应用时原子创建本批独立 taxel，随后丢弃范围、行列和扫描参数。没有基于隐藏区域对象的“重新生成”；需要调整时多选并移动、重新投影或删除后重新批量添加。

当前 taxel 数组行序就是输出顺序，不维护另一套输出索引。批量复制到另一侧只复用尚在界面的生成参数，必须重新拾取表面、绑定关节和分配索引，不镜像几何。

文件中的输出顺序由 taxel 数组行序表达；内部标识和临时编号不写入文件，重新打开后按文件行序建立编号。

选择提供三种入口：单击并用 Ctrl/Cmd 增减；框选画面范围；输入 taxel 索引范围或列表。框选模式暂时接管拖动，退出后恢复相机操作；默认只选择当前侧、可见且未被网格遮挡的点，“穿透选择”显式包含背面。多选属性中的不同值显示“多个值”，批量修改只作用于填写字段。

选中后可编辑绑定根中的位置、法线翻转、切线旋转和 taxel 索引。表面移动在指定表面重新拾取，自由移动标记为手工来源。高级折叠项提供 XYZW 四元数，不要求正常用户填写。列表每页 25 条，不创建逐 taxel 常驻控件。

### 1.7 绑定页

只有一套操作：编辑所选 taxel 的四个索引和权重槽，通常选择一至四个 segment。没有独立的刚性、两段过渡或手工模式。原始槽顺序始终保留，不按 segment 合并重复槽；权重在明确编辑时按下述规则更新。

右侧显示四槽，每槽包含 segment 下拉框、原始索引、颜色标记、0–1 权重滑块和独立的权重数值框；上方显示选中 taxel 数量和权重和。缺失、越界及重复索引仍显示原值。新选关节默认等分。通过滑块或数值框调整一个权重时，先替换该槽的值，再对该 taxel 的全部四槽执行 `w_k ← w_k / Σ_i w_i`，使权重和为 1；被调整槽的最终值也参与归一化。拒绝负权重输入；四槽含非有限值、负值或总和为零时，拒绝本次编辑、保留原行并显示原因。载入、显式记录接收值、检查、保存和导出保留原始权重，不自动归一化、累加或修复；其中的非法值仍须报告为错误。

选择或调整某槽权重时，通过颜色高亮对应关节、关联链接和当前 taxel，显示关节到 taxel 的连线，线宽或亮度表达权重。多选时保留黄色选择轮廓，以当前关节的权重给所选点着色；混合值显示为混合，不伪造统一值。批量调整对每个 taxel 独立执行四槽归一化。

“运动预览”使用草稿数组和 URDF 滑块检查绑定，草稿仍可编辑。“返回绑定姿态”结束预览。单段可显示运动坐标轴；多段只显示位置与力模长，不显示协议未定义的世界空间力箭头。偏离网格的点保持可见，不重新吸附来掩盖误差。

### 1.8 实时页

页面从上到下包含以下区块与控件：

**数据源（每侧分别配置）**

- 触觉 collection_id 输入框。By default "tactile".
- HandTracker 左手／右手输出选择.
- JointState 输入框 (optional, by default disabled).

**URDF 驱动**

实时 URDF 只能由启用的 JointState 输入驱动；未启用时隐藏 URDF，仅显示手骨架和 taxel。HandPose 不驱动 URDF，只提供实际手骨架和实时 taxel LBS 当前段位姿。EDIT/PREVIEW 中的 q0 编辑和手动运动预览不受此实时规则影响。

- JointState 映射 JSON 文本框：`{源关节名: [URDF 自由关节, 比例, 偏移]}`，整表应用。
- 各关节的原始输入值、换算值、限位与映射状态。

**连接与播放**

- 连接／断开按钮。
- 播放／暂停及返回 q0 显示按钮；这些操作不改变草稿的可编辑性。
- 记录 metadata 按钮，提供分组勾选项，默认全部选中。每组字段一并选择：
  - Skeleton binding poses：`joint_ids`、`segment_bind_poses`。
  - Taxel binding poses：`taxel_bind_poses`。
  - Skinning：`taxel_segment_indices`、`taxel_segment_weights`。
- 显式记录前先按第 1.1 节的相同 schema JSON 格式将当前 metadata 数值备份到临时位置，再替换所选分组；日志显示备份路径。
- 三维可视化 metadata 来源切换：接收的 metadata／用户草稿，默认接收来源。所有页签中的 metadata 数值始终来自草稿。
- 接收 metadata 缺失或无效时，HandPose 与 JointState 继续更新，三维 metadata 临时使用草稿并显示原因。
- 连接状态。各数据源的接入类型、配置标识、有效性和本地读取时间。
- 工具操作事件、警告和错误日志，包括替换前的草稿备份路径。

实时接收和三维播放不自动替换草稿。每次显式记录仅以接收值替换所选分组，不合并旧值；替换后即使产生引用或数组长度错误，也保留结果并在对应区域显示原因。

metadata 与力值的实际对应关系由生产端和使用者负责。工具按当前 metadata 与力数组的协议行序显示，执行协议结构、数量及有效性检查；不维护独立的源行映射或对齐基线，不为 metadata 编辑重新排列力值。切换三维 metadata 来源不建立或修复它与力流的对应关系。

### 1.9 导出、检查与状态跳转

导出页选择当前侧或全部，提供“本地检查”“SDK 检查”和“导出”。问题按 error、warning、info 显示，点击后定位页面和实体。error 阻止正式导出；当前会话中的近似来源产生 warning，显示在检查结果中，不引入逐项确认门槛。

本地与 SDK 状态分开。SDK 探测成功只表示环境可用；未完成元数据构造不能显示官方通过。JSON 导出只要求本地无 error；SDK 导出还要求官方构造与大小检查通过。导出目录仅包含各侧 metadata JSON 及用户选择的 SDK 二进制；检查结果输出到界面或 CLI，不作为附属文件保存。

| 状态 | 显示与可用操作 | 跳转 |
| --- | --- | --- |
| EDIT | q0 下的 URDF、segments 和 taxel；允许全部静态编辑 | 运动预览进入 PREVIEW；连接并播放进入 LIVE |
| PREVIEW | 滑块驱动模型与 LBS；表单始终编辑草稿 | 返回绑定姿态恢复 EDIT |
| LIVE | 连续更新三维模型、骨架和力值；三维 metadata 来源可切换，所有表单仍显示并编辑草稿 | 暂停进入 PAUSED |
| PAUSED | 冻结实时三维层；接收继续；表单和草稿操纵器仍可编辑 | 恢复取最新帧；返回 q0 显示进入 EDIT |
| EDIT 且已连接 | q0 编辑；后台只保留最新数据，不覆盖文档 | 播放重新进入 LIVE |
| 数据中断 | 保留最后画面并标记失效；草稿继续可编辑，不能把旧力视为当前值 | 可回 EDIT；重连后重新检查数据有效性 |

所有模式下，数值表单及草稿操纵器都编辑 q0 下的手根坐标，不反解实时混合蒙皮，也不把实时位姿写回草稿。三维操纵器针对明确选中的草稿实体；接收 metadata 的三维图形只读，不能因拾取它而替换表单。PAUSED 只冻结实时三维层，草稿操纵器随编辑更新；恢复后使用最新帧及当前草稿重新计算。播放、暂停、选择、相机和实时帧不修改 Document。

本地保存允许未完成的 metadata 数值，仍使用第 1.1 节的相同字段结构；“编辑草稿，尚非有效 metadata”仅为界面提示，不写入文件。正式导出必须完整，校验作为独立步骤，不引入另一种序列化格式。导出使用启动时的 metadata 数值快照，继续编辑不改变导出内容，界面或 CLI 结果显示对应 revision。目标目录非空时要求新目录；全部手侧成功后才发布数值文件。

### 1.10 命令行

GUI 和 CLI 共用计算、检查与导出，通过 Python 模块入口启动。以下命令使用调用者选定的 Python 环境，从工具目录执行；`python` 指该环境的解释器，工具不替换它：

```bash
cd scripts/tactile_metadata
python -m tactile_metadata_editor.cli gui --urdf glove.urdf --sides left
python -m tactile_metadata_editor.cli gui --urdf hands.urdf --sides both --left-metadata left.metadata.json --right-metadata right.metadata.json
python -m tactile_metadata_editor.cli gui --metadata left.metadata.json --sides left
python -m tactile_metadata_editor.cli check left.metadata.json
python -m tactile_metadata_editor.cli check left.metadata.json --sdk
python -m tactile_metadata_editor.cli export left.metadata.json --sides left --format sdk --out package
```

GUI 不带参数时进入启动页，默认监听 0.0.0.0:8080，可用 --port 修改。JSON 不携带手侧或 URDF 配置：GUI 通过界面或 `--sides` 明确手侧，CLI export 通过 `--sides left/right/both` 分配输入文件，手侧重复明确拒绝；纯数值 CLI 检查不需要手侧。检查结果输出到界面或 CLI 标准输出，不提供 `--report` 文件输出。实时连接在当前 GUI 会话中配置；`--demo` 使用内置合成数据验证实时交互，不是额外文件导入。

## 2 依赖

| 项目 | 决策 | 原因与用途 |
| --- | --- | --- |
| 语言 | Python >=3.11,<3.14 | 与 SDK 生态衔接，工具本身无需原生编译。 |
| 浏览器 GUI | viser==1.1.1 | 复用三维客户端、表单、框选、射线事件和控制器。 |
| URDF | yourdfpy==0.0.60 | 资源解析、链接图、关节配置和 FK。 |
| 网格处理 | trimesh==5.1.1 | URDF 网格资源、法线、连通分量和射线。 |
| 数值运算 | numpy>=2,<3 | 批量变换、数组校验和 LBS 位置计算。 |
| 查询加速 | rtree>=1.3,<2 | 三角形空间查询，不要求 Embree。 |
| SDK 集成 | 启动工具的 Python 环境中可用的 Isaac Capture SDK | 官方编码以及实时 tracker/session；复用该环境中的 SDK，离线流程不导入 SDK。 |
| 本地通信 | Python 标准库 subprocess、socket、json | SDK 隔离与实时数据传输，不增加消息中间件。 |
| 依赖维护与运行 | Isaac Capture 现有环境依赖声明及 Python 模块入口 | 补齐缺失库，复用启动工具的环境；不建立工具专用项目、虚拟环境或锁文件。 |
| 开发检查 | 仓库已有检查，按用户指定范围执行 | 不新增测试文件或测试依赖，不运行越出工作目录或可能改写其他目录的检查。 |

将缺失库及所需版本约束加入 Isaac Capture 现有的 [`src/core/python/requirements*.txt`](../../src/core/python/) 对应依赖分组，并同步[根 pyproject.toml](../../pyproject.toml) 等对应包依赖声明。已有依赖按所需版本复用或补充约束，不另建工具专用 `pyproject.toml`、`uv.lock` 或 `.venv`。依赖准备使用调用者选定的同一 Python 环境；工具启动时不自动安装库、创建环境或构建 SDK。

版本依据为 [Viser](https://pypi.org/project/viser/1.1.1/)、[yourdfpy](https://pypi.org/project/yourdfpy/0.0.60/) 和 [trimesh](https://pypi.org/project/trimesh/5.1.1/) 发布页。实施时核对共享依赖声明和实际安装版本，并验证模型加载。Viser 框选接口以 [v1.1.1 源码](https://github.com/nerfstudio-project/viser/blob/v1.1.1/src/viser/_scene_api.py) 为准，不把 main 文档新增接口当成指定版本保证。

离线编辑不依赖 ROS、CUDA、OpenXR 会话或模拟器。官方检查、SDK 二进制导出和实时接收使用启动工具的 Python 环境中可导入的 SDK；实时接收还需要对应 runtime/设备，当前 live OpenXR 路径需要可用 HMD system，依据[设备文档](../../docs/source/device/tactile/index.rst)。SDK 仅在以 `sys.executable` 启动、继承启动环境的本地子进程中导入。不可用时显示原因，离线编辑仍可运行；不查找旧构建目录、不切换解释器、不提供 SDK 路径配置，也不要求实现机器预先具备未来用户的运行环境。


## 3 给 agent 的范围规定风格规范和 restriction

### 3.1 范围与工作规范

实现第一节定义的完整工具，工具代码及模型示例位于 scripts/tactile_metadata/。允许为本工具补齐 Isaac Capture 现有环境依赖声明及对应包元数据；不建立工具专用环境。不改 schema、golden 或无关子系统；真实冲突提供代码证据并同步规格。设计工具架构以确保well structured and functionally correct，并且concise without redundant codes, strictly limited to the explicitly described scope。制定的实现计划行数预估必须小于2000行（不包括文档）。

修改前列出目录并读取全部适用 AGENTS.md，保护已有改动。本地检查，不连接远程计算机，不自动提交或推送。不把本机 SDK/构建/资源绝对路径硬编码到工具或可移植文件。

领域类型集中，GUI/CLI 共享数组准备、检查和导出；按模块契约实现，不另造通用框架。遵守仓库日志、SPDX、简短注释、格式化和 Mandatory learning loop。文件输入范围、术语和批量操作生命周期是产品约束，不能为实现便利恢复被移除入口。

### 3.2 实施顺序与交付

制定实施计划后，按合理的逻辑顺序执行实施。

交付复用启动 Python 环境的工具、README中包括使用工具的step to step instruction，以本目录现有model/文件夹内urdf为示例、面向开发者的技术说明（解释模块和数据流等）。不交付工具专用环境或锁文件，不要求新增 URDF、网格或测试文件，也不新增测试依赖。合成实时源可在无 SDK/硬件时运行，实际 live 接收仍须实现，不能用合成数据代替。README 逐步说明在已有 Isaac Capture 环境中准备依赖并启动工具，以及离线编辑、schema JSON 继续编辑、四槽权重归一化、taxel 输出顺序和实时播放/暂停/编辑/恢复。
