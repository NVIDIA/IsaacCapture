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

输出为可继续编辑的 metadata JSON、纯数值 metadata JSON、taxel 对应表、检查报告及可选的 SDK 二进制数据包。继续编辑所需的 URDF 路径、映射和校准信息保存在同一 JSON 的可选 editor 扩展中；没有独立的工程文件输入。无 editor 扩展的数值 JSON 仍须支持。

实时输入首版接入 TactileDataArray/TactileMetadata、HandPose 和可选 JointStateOutput。真实设备协议接入、原始 ADC/压力到牛顿的转换、设备控制及物理标定不在工具内完成。工具只接收现有 Isaac Capture 数据；编辑、保存和导出均不向 runtime 或生产设备发送 metadata、控制命令或其他数据。

### 1.2 窗口布局与显示

一个浏览器窗口包含三维画布和右侧标准 medium 宽度面板。面板顶部显示文件、已固定的 URDF 手侧及当前模式；下方是模型、Segments、布局、绑定、实时、导出六个页签。长表单滚动，复用 Viser 客户端；撤销重做所需的浏览器事件桥接见 4.3，不维护独立前端。

```text
┌──────────────────────────────────┬──────────────────────────────┐
│                                  │ URDF 名称；手侧：双手（只读） │
│        三维模型与手骨架           │ Metadata 文件 *   保存 撤销重做│
│                                  │ 编辑对象：[左手] [右手]       │
│ taxel、选中关节、法线和切线       │ 模式：编辑 / 预览 / 实时 / 暂停│
│                                  ├──────────────────────────────┤
│ 相机：旋转 平移 缩放 重置         │ 模型 Segments 布局 绑定 实时 导出│
│ 选择：单选 / 框选 / 穿透选择      │ 当前页控件与选择属性           │
│                                  ├──────────────────────────────┤
│ Info：草稿错误状态及具体原因      │ 连接和数据状态               │
└──────────────────────────────────┴──────────────────────────────┘
```

模型为中性色。编辑时 taxel 按主影响关节着色，无绑定为灰色，选中轮廓为黄色，错误轮廓为红色。实时触觉用力向量模长映射蓝至红，默认范围 0–30 N，可调整上限，超限饱和；无有效数据为灰色，不能显示成零力。属性面板显示 fx、fy、fz 和模长，单位 N。

球形标记大小只影响显示，不改变 sensing location。选中 taxel 显示 +Z=外向法线 N、+X=传感器切线 T、+Y=N×T；X×Y=Z。双手使用同一右手坐标约定。多关节 LBS 动画只保证位置，不把混合旋转解释为协议定义的唯一方向。

六页表单始终显示并编辑 Document 中的草稿值，包括 PREVIEW、LIVE、PAUSED 及连接中断时。三维视图的 metadata 来源切换不改变表单数据；接收值只有通过“记录 metadata”显式替换所选草稿分组后才成为表单值。力值和连接状态是单独的只读实时信息。

允许草稿包含错误。出错字段及所属分组显示错误状态和原因，三维视图下方的紧凑 Info 区汇总当前手侧、分组、实体和原因，并可定位对应字段。保留能显示的有效几何；缺失或无效的位姿不以假位置绘制。错误不阻止继续编辑、撤销、重做或替换数据，只阻止正式导出。

### 1.3 文件、URDF 与左右手

启动页提供“载入 URDF”和“打开 metadata JSON”。载入 URDF 时选择左手、右手或双手，完成后只读；替换 URDF 或改变手侧需要创建新文档，先处理未保存内容。URDF 不携带标准手侧字段，该选择是用户对模型的明确声明。

一个会话使用一个 URDF。双手选择要求模型实际包含左右手结构，随后分别指定腕部链接和关联链接；不能把单手模型自动镜像成双手。独立左右 URDF 分别打开文档。双手文档中的左右 metadata 独立，但共享同一 URDF 和绑定关节配置；共享关节的修改会明确列出受影响手侧。

顶部“编辑对象”只选择当前要编辑的 metadata，不修改模型手侧。单手不显示切侧控件；双手可以显示当前侧或双侧，另一侧 taxel 只读。任何显示排布偏移均不参与元数据计算。两侧数量与输出排列可以不同，各自的 taxel 索引从 0 开始。

新建 metadata 不要求输入预期数量，界面从当前 taxel 记录计算 N。检查与实时数据比较时才核对数量。全工具使用“taxel 数量”“taxel 索引”和“输出顺序”，不以 channel 表示 taxel。

打开无 editor 扩展的 JSON 时，用户将其分配到当前 URDF 的左侧或右侧；不能从关节名称推断手侧。导入保留原 segment 和 taxel 行顺序。仅有 JSON 时可以查看和修改数值；关联 URDF、腕部及关节后才启用表面操作和模型动画。

模型、JSON 和 package 资源均使用本地路径，不要求逐个上传网格。打开新文档、替换一侧 metadata 或退出时，有未保存内容则提供保存、放弃、取消。双手分别保存两个 metadata JSON；“保存全部”处理两侧，失败不得将任一侧误标为已保存。

### 1.4 模型页

页面提供 URDF 路径、package:// 的本地目录映射、资源状态、链接可见性和透明度。可检查模型树、关节类型与范围；缺失资源列出具体路径，不从网络寻找模型。

用户为每侧选择腕部链接，并拾取腕中心 O、朝前臂的参考 F、朝手背的参考 D，建立来源根坐标系。双手分别校准，界面显示两套参考轴，不按左/右自动变号。关联链接限制该侧拾取范围；共享掌臂链接可以作为参考，但另一侧表面不能成为默认拾取对象。

### 1.5 Segments 页

页面位于布局之前，包含“绑定姿态”和“Segments”两组控件。

绑定姿态组提供 URDF 自由关节滑块、数值输入和“设为绑定姿态”。fixed 不可编辑，mimic 跟随主关节，限位始终可见。此姿态 q0 是静态布局和 bind poses 的共同参考；实时角度及运动预览角度不会自动写入 q0。

Segments 组从 hand.fbs 的 HandJoint 枚举选择需要的关节，排除 NUM_JOINTS。表格显示语义名称、输出行号、关联 URDF 链接和参考系状态。合法 metadata 每侧同名语义只出现一次，不要求完整骨架，WRIST 使用根恒等位姿；草稿违反这些要求时显示错误并保留值。选择一行会高亮关节球、坐标轴和关联链接。

新建 segment 时选择语义和链接，在模型上定位关节中心及参考方向，通过控制器调整位置与扭转。已有 metadata 的 B_j 已给出坐标：选择目标链接后计算固定局部修正，保持原 B_j，而不是把数值吸附到 URDF 原点。

用户可以调整有序 segment 列表。显式重排通过行号置换更新已有合法 skinning 索引，保持其稳定实体引用；语义枚举值不参与行号运算。允许删除仍被引用的 segment；未解决的引用留在草稿中，绑定页和 Info 区显示原因，不隐式重绑。

设置新 q0 或修改腕部参考系时，先预览模型和附件变化；现有 B_j、taxel 的绑定根位姿和权重保持数值不变，确认后重算关节局部附件并使检查过期。实际 B_j 的修改通过关节参考系控制器完成。工具不隐式重投影、蒙皮重绑定或修改生产端 metadata。偏离模型的点保留并提示用户调整。

### 1.6 布局页

页面提供“批量添加”“放置单点”和所选 taxel 的属性。没有区域名、区域列表或持久化批次；批量添加只是一次生成操作，导入 metadata 与新建结果拥有相同编辑能力。

批量添加只有一种方式：输入范围和行列数量。操作依次为：

1. 选择允许投影的链接和表面，输入 rows、cols。
2. 拾取首点中心 O、首行末点中心 C、首列末点中心 R。显示列跨度、行跨度及推导间距；可直接修改跨度数值，仍属于同一个生成模式。
3. 选择 row-major、column-major、row-snake 或 column-snake、行列反向和起始 taxel 索引，预览后应用或取消。

范围以首末 taxel 中心定义，不是传感片外边缘。单行只需 O/C，单列只需 O/R，单点只需 O 和切线方向，避免除以零。可在预览中排除实际不存在的单元；被排除单元不产生 taxel，投影失败单元则保留为有问题的候选，不能混为一谈。

预览显示位置、数量和输出索引，错误标红。应用作为一个可撤销事务创建独立 taxel，随后丢弃范围、行列和扫描参数。没有基于隐藏区域对象的“重新生成”；需要调整时多选并移动、重新投影或删除后重新批量添加。

默认从尚未使用的连续索引区间分配新点。移动、重投影和删除不重排已有输出；“重排所选 taxel”显示旧→新索引对应表后应用，检查要求最终覆盖 0..N-1。批量复制到另一侧只复用尚在界面的生成参数，必须重新拾取表面、绑定关节和分配索引，不镜像几何。

选择提供三种入口：单击并用 Ctrl/Cmd 增减；框选画面范围；输入 taxel 索引范围或列表。框选模式暂时接管拖动，退出后恢复相机操作；默认只选择当前侧、可见且未被网格遮挡的点，“穿透选择”显式包含背面。多选属性中的不同值显示“多个值”，批量修改只作用于填写字段。

选中后可编辑绑定根中的位置、法线翻转、切线旋转和 taxel 索引。表面移动在指定表面重新拾取，自由移动标记为手工来源。高级折叠项提供 XYZW 四元数，不要求正常用户填写。列表每页 25 条，不创建逐 taxel 常驻控件。

### 1.7 绑定页

只有一套操作：编辑所选 taxel 的四个索引和权重槽，通常选择一至四个 segment。没有独立的刚性、两段过渡或手工模式。原始槽顺序和数值始终保留，不按 segment 合并重复槽。

右侧显示四槽，每槽包含 segment 下拉框、原始索引、颜色标记、0–1 滑块和权重数值框；上方显示选中 taxel 数量和权重和。缺失、越界及重复索引仍显示原值。新选关节默认等分；拖动滑块时其余槽按现有比例分配剩余权重，剩余总和为零则等分。直接修改数值只写入该字段，允许非法值并显示原因。分配规则只用于明确的用户操作；载入、接收值替换、检查和导出不归一化、不累加或修复权重。

悬停或拖动权重时，高亮对应关节、关联链接和当前 taxel；其余模型淡化，显示关节到 taxel 的连线，线宽或亮度表达权重。多选时保留黄色选择轮廓，以当前关节的权重给所选点着色；混合值显示为混合，不伪造统一值。批量调整对每个 taxel 分别执行剩余权重分配。

“运动预览”使用草稿数组和 URDF 滑块检查绑定，草稿仍可编辑。“返回绑定姿态”结束预览。单段可显示运动坐标轴；多段只显示位置与力模长，不显示协议未定义的世界空间力箭头。偏离网格的点保持可见，不重新吸附来掩盖误差。

### 1.8 实时页

页面从上到下包含以下区块与控件：

**数据源（每侧分别配置）**

- 触觉 collection_id 输入框。By default "tactile".
- HandTracker 左手／右手输出选择.
- JointState 输入框 (optional, by default disabled).

**URDF 驱动**

实时 URDF 只能由启用的 JointState 输入驱动；未启用时隐藏 URDF，仅显示手骨架和 taxel。HandPose 不驱动 URDF，只提供实际手骨架和实时 taxel LBS 当前段位姿。EDIT/PREVIEW 中的 q0 编辑和手动运动预览不受此实时规则影响。

- JointState 映射表：源关节名、URDF 自由关节、比例、偏移。
- 各关节的原始输入值、换算值、限位与映射状态。

**连接与播放**

- 连接／断开按钮。
- 播放／暂停及返回 q0 显示按钮；这些操作不改变草稿的可编辑性。
- 记录 metadata 按钮，提供分组勾选项，默认全部选中。每组字段一并选择：
  - Skeleton binding poses：`joint_ids`、`segment_bind_poses`。
  - Taxel binding poses：`taxel_bind_poses`。
  - Skinning：`taxel_segment_indices`、`taxel_segment_weights`。
- 显式记录前先将当前草稿保存到临时位置，再替换所选分组；日志显示备份路径。
- 三维可视化 metadata 来源切换：接收的 metadata／用户草稿，默认接收来源。所有页签中的 metadata 数值始终来自草稿。
- 连接状态。各数据源的接入类型、配置标识、有效性和本地读取时间。
- 工具操作事件、警告和错误日志，包括替换前的草稿备份路径。

实时接收和三维播放不自动替换草稿。每次显式记录仅以接收值替换所选分组，不合并旧值；替换后即使产生引用或数组长度错误，也保留结果并在对应区域显示原因。

### 1.9 导出、检查与状态跳转

导出页选择当前侧或全部，提供“本地检查”“SDK 检查”和“导出”。问题按 error、warning、info 显示，点击后定位页面和实体。error 阻止正式导出；近似来源产生 warning，进入报告而不引入逐项确认门槛。

本地与 SDK 状态分开。SDK 探测成功只表示环境可用；未完成元数据构造不能显示官方通过。纯 JSON 数据包只要求本地无 error；SDK 数据包还要求官方构造与大小检查通过。

| 状态 | 显示与可用操作 | 跳转 |
| --- | --- | --- |
| EDIT | q0 下的 URDF、segments 和 taxel；允许全部静态编辑 | 运动预览进入 PREVIEW；连接并播放进入 LIVE |
| PREVIEW | 滑块驱动模型与 LBS；表单始终编辑草稿 | 返回绑定姿态恢复 EDIT |
| LIVE | 连续更新三维模型、骨架和力值；三维 metadata 来源可切换，所有表单仍显示并编辑草稿 | 暂停进入 PAUSED |
| PAUSED | 冻结实时三维层；接收继续；表单和草稿操纵器仍可编辑 | 恢复取最新帧；返回 q0 显示进入 EDIT |
| EDIT 且已连接 | q0 编辑；后台只保留最新数据，不覆盖文档 | 播放重新进入 LIVE |
| 数据中断 | 保留最后画面并标记失效；草稿继续可编辑，不能把旧力视为当前值 | 可回 EDIT；重连后重新核对基线与对应 |

所有模式下，数值表单及草稿操纵器都编辑 q0 下的手根坐标，不反解实时混合蒙皮，也不把实时位姿写回草稿。三维操纵器针对明确选中的草稿实体；接收 metadata 的三维图形只读，不能因拾取它而替换表单。PAUSED 只冻结实时三维层，草稿操纵器随编辑更新；恢复后使用最新帧及当前草稿重新计算。播放、暂停、选择、相机和实时帧不进入撤销历史。

Ctrl+Z 撤销，Ctrl+Shift+Z 重做；macOS 同时支持 Cmd 对应组合。字段数值修改、taxel/segment 关节/URDF 链接移动及接收 metadata 替换均在历史中，行为见 4.3。

本地保存允许未完成草稿，并明确标记“编辑草稿，尚非有效 metadata”；正式导出必须完整。导出使用启动时快照，继续编辑不改变导出内容，结果显示对应 revision。目标目录非空时要求新目录；全部手侧成功后才发布数据包。

### 1.10 命令行

GUI 和 CLI 共用计算、检查与导出，通过 Python 模块入口启动。以下命令使用调用者选定的 Python 环境，从工具目录执行；`python` 指该环境的解释器，工具不替换它：

```bash
cd scripts/tactile_metadata
python -m tactile_metadata_editor.cli gui --urdf glove.urdf --sides left
python -m tactile_metadata_editor.cli gui --urdf hands.urdf --sides both --left-metadata left.metadata.json --right-metadata right.metadata.json
python -m tactile_metadata_editor.cli gui --metadata left.metadata.json
python -m tactile_metadata_editor.cli check left.metadata.json
python -m tactile_metadata_editor.cli check left.metadata.json --sdk
python -m tactile_metadata_editor.cli export left.metadata.json --format sdk --out package
```

GUI 不带参数时进入启动页，默认监听 127.0.0.1:8080，可用 --port 修改。没有 editor 配置的 JSON 通过界面或 --sides 明确手侧，纯数值 CLI 检查不需要手侧。check 可加 --report；export 可重复传入 metadata JSON 处理双手，手侧重复或 URDF 配置冲突明确拒绝。实时连接在 GUI 中配置；--demo 使用内置合成数据验证实时交互，不是额外文件导入。

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
| 本地通信 | Python 标准库 subprocess、socket、json、http.server | SDK 隔离、实时数据传输及浏览器事件桥接，不增加消息中间件。 |
| 依赖维护与运行 | Isaac Capture 现有环境依赖声明及 Python 模块入口 | 补齐缺失库，复用启动工具的环境；不建立工具专用项目、虚拟环境或锁文件。 |
| 开发检查 | 仓库已有检查，按用户指定范围执行 | 不新增测试文件或测试依赖，不运行越出工作目录或可能改写其他目录的检查。 |

将缺失库及所需版本约束加入 Isaac Capture 现有的 [`src/core/python/requirements*.txt`](../../src/core/python/) 对应依赖分组，并同步[根 pyproject.toml](../../pyproject.toml) 等对应包依赖声明。已有依赖按所需版本复用或补充约束，不另建工具专用 `pyproject.toml`、`uv.lock` 或 `.venv`。依赖准备使用调用者选定的同一 Python 环境；工具启动时不自动安装库、创建环境或构建 SDK。

版本依据为 [Viser](https://pypi.org/project/viser/1.1.1/)、[yourdfpy](https://pypi.org/project/yourdfpy/0.0.60/) 和 [trimesh](https://pypi.org/project/trimesh/5.1.1/) 发布页。实施时核对共享依赖声明和实际安装版本，并验证模型加载。Viser 框选接口以 [v1.1.1 源码](https://github.com/nerfstudio-project/viser/blob/v1.1.1/src/viser/_scene_api.py) 为准，不把 main 文档新增接口当成指定版本保证。

离线编辑不依赖 ROS、CUDA、OpenXR 会话或模拟器。官方检查、SDK 二进制导出和实时接收使用启动工具的 Python 环境中可导入的 SDK；实时接收还需要对应 runtime/设备，当前 live OpenXR 路径需要可用 HMD system，依据[设备文档](../../docs/source/device/tactile/index.rst)。SDK 仅在以 `sys.executable` 启动、继承启动环境的本地子进程中导入。不可用时显示原因，离线编辑仍可运行；不查找旧构建目录、不切换解释器、不提供 SDK 路径配置，也不要求实现机器预先具备未来用户的运行环境。

## 3 代码架构

### 3.1 分层与数据流

```text
ui / cli
  ├─ 编辑命令 → state → Document
  ├─ 生成请求 → layout ← model.surface_snapshot(q0)
  │                └─ Candidate → state.dispatch
  ├─ 数组准备 → compiler ← model + profiles
  │                └─ CompiledDevice → mappings.evaluate → 场景
  ├─ 检查/导出 → checks → sdk → sdk_worker → export
  └─ 实时连接 → live → live_worker → Isaac Capture trackers/session
                    └─ LiveSnapshot → drivers → 场景
                         └─ 显式导入 → Candidate → state.dispatch

document 读写 metadata JSON；domain 定义共享类型。
state 的工程快照和 live 的只读数据快照互相独立。
```

Document 是唯一可编辑来源。生成器只返回候选，state 负责应用；接收 metadata 通过 state 单向替换选定草稿分组。compiler 仅准备数组：按明确的输出索引排列、检查引用并转换协议 dtype，不编译源码、不按关节累加、不重写权重。预览、检查、保存及导出使用同一输出规则；所有表单直接读取草稿，不读取接收快照或旧的检查结果。

### 3.2 模块契约

| 模块 | 公开接口 | 输出与边界 |
| --- | --- | --- |
| domain.py | Document、Device、Segment、Taxel、Issue、Candidate、LiveSnapshot 等 | 只定义类型，不导入 GUI、SDK、URDF。 |
| document.py | load_metadata(path)、save_metadata(snapshot,path)、import_values(values,side) | 解析/保存 JSON 与 editor 扩展；错误草稿可保存，文件写入保持原子性。 |
| state.py | dispatch(command)、snapshot()、undo()、redo() | 唯一编辑入口，维护 revision、dirty 与显示模式。 |
| model.py | load_urdf(asset)、link_pose(link,q)、surface_snapshot(q) | 链接局部网格、FK 和只读几何。 |
| layout.py | generate_batch(spec,snapshot)、place_taxel(hit,spec) | 返回 Candidate，不保留 Region 或批次实体。 |
| profiles/hand.py | resolve(device,model,q)、source_payload(segments) | 腕根、有序关节姿态和 HandSkeletonRef。 |
| mappings/lbs.py | edit_weights(...)、validate_slots(...)、evaluate(...) | 显式权重操作、原始四槽校验和位置预览，不按关节汇总。 |
| compiler.py | compile_device(snapshot,device_id,model) | 只读协议数组 CompiledDevice 或 CompileFailure(issues)，不修改草稿。 |
| checks.py | check_device(snapshot,device_id,model,use_sdk=False) | 数组或空值、issues、独立检查状态。 |
| sdk.py / sdk_worker.py | probe()、encode(values,timeout=30) | 使用当前解释器的短生命周期 SDK 子进程与文件协议。 |
| live.py / live_worker.py | connect(config)、latest()、disconnect() | 长生命周期本地 SDK 接收进程；只读 LiveSnapshot。 |
| drivers.py | map_joints(config,sample)、current_segments(...)、view_root(...) | 数据到 URDF、当前段及视图根的变换，不修改 bind 数据。 |
| export.py | export_package(snapshot,device_ids,options) | 一致快照的完整数据包。 |
| ui.py / cli.py | 界面事件和 gui/check/export 入口 | 只调用上述接口，不重复几何和数值逻辑；ui.py 接收浏览器桥接事件并投递状态队列。 |
| ui_bridge.js | 浏览器手势、数值确认及撤销重做快捷键 | 在现有 Viser 页面内运行，仅发送本地编辑事件，不访问 Document 或 SDK。 |

Snapshot 不可变。Candidate 包含 base_revision、记录增删改和 issues。CompiledDevice 包含有序数组、segment ID→行号和输出行→taxel ID。Issue 包含 severity、code、device_id、entity_id、field、message。

数组准备失败返回可定位的 CompileFailure，checks 返回 compiled=null，不能以假数组掩盖缺字段；这不阻止保留、显示和编辑错误草稿。local_status 与 sdk_status 分开。LiveSnapshot 包含连接 epoch、poll 序号、本地读取时间、metadata 基线及独立有效性标记；不与 Document 共用可写数组。

### 3.3 扩展边界

profile_registry 按 SegmentSource 类型选择设备语义；mapping_registry 按 GeometryMapping 类型选择数组与变形逻辑。首版分别只注册 HandSkeletonRef、LBS。通用设备、taxel、事务、选择与文件容器不直接依赖 HandJoint 或手侧。

实时输入另有类型适配接口，将 SDK 对象复制成工具拥有的只读值。新增设备来源时扩展 profile 和对应驱动；增加映射算法时扩展 mapping，不把未来设备预先伪装成手套。未知类型或版本拒绝处理并保留原文件，不建立热加载插件系统。

### 3.4 运行与目录

GUI 进程包含单一状态循环、几何/检查工作线程及实时 socket 接收线程。SDK 编码与实时接收分别在本地子进程中运行，不让包级原生导入影响 GUI。一个客户端有编辑权，其余只读。

工具作为仓库内 Python 模块运行，依赖随 Isaac Capture 环境维护；进程隔离不建立另一套 Python 或 SDK 环境。

```text
scripts/tactile_metadata/
  AGENTS.md
  DESIGN.md
  IMPLEMENTATION_PROMPT.md
  README.md
  tactile_metadata_editor/
    __init__.py
    domain.py
    document.py
    state.py
    model.py
    layout.py
    compiler.py
    checks.py
    sdk.py
    sdk_worker.py
    live.py
    live_worker.py
    drivers.py
    export.py
    ui.py
    ui_bridge.js
    cli.py
    profiles/hand.py
    mappings/lbs.py
  model/
  examples/
```

## 4 每个模块内部的逻辑实现和 convention

### 4.1 领域数据与坐标

Document 包含可空 model、有序 devices 和共同 bind_configuration。模型配置保存 URDF 路径、资源映射、hash、局部链接调整及不可变 sides；每侧 Device 包含 profile、mapping、segments、taxels、skinning 和可选 application_binding。总数 N 从 taxel 位姿记录推导，没有预先填写的数量字段。

实体使用 UUID；Taxel 保存 id、可空 taxel_index、绑定根中的可空 position_m/orientation_xyzw、可空 surface_attachment、generation_issue、provenance 及可选输入对应。skinning 独立保存原始 taxel_segment_indices 行和 taxel_segment_weights 行，按 taxel 输出索引关联；不在 Taxel 中另存一套汇总权重。错误草稿可保留不匹配的行数、缺失或越界值；不能为凑齐行数生成假位姿、截断或补造接收值。没有 Region、region_id、持久化 batch 或阵列 cell。

稳定 ID 标识实体；四槽数值仍是编辑和导出的权威。显式重排 segment 时，state 对合法索引应用同一行号置换以保持实体引用；显式重排 taxel 时一并移动其 skinning 行。接收分组替换直接保留接收行号，不自动匹配语义、合并槽或重绑；留下的错误由检查报告。

Segment 保存 id、frame_ref 和局部 mount pose。frame_ref 为 root 或 link:<name>；未关联 URDF 的导入段保留原根位姿，关联后改存链接局部修正。Taxel 始终以绑定根位姿为权威数据；表面附件仅用于追踪资源和显式重投影，不反向驱动已存 taxel。

位置用米，角度用弧度，单位四元数为 XYZW。T_A_B 把 B 中的点变到 A；M 是 URDF 根，W 是腕根，J 是 segment，X 是 taxel。GUI 的毫米/度及 Viser 的 WXYZ 只在边界转换。缺失位姿用 null，不以零位置、单位四元数或 NaN 冒充有效几何。

### 4.2 metadata JSON 与继续编辑

数值格式保留 format=isaaccapture.tactile-values、version=1；它是工具对 TactileMetadata 的 JSON 表达，不是 FlatBuffers 原生 JSON。完整文件示例：

```json
{
  "format": "isaaccapture.tactile-values",
  "version": 1,
  "segment_source": {"type": "HandSkeletonRef", "joint_ids": ["WRIST", "INDEX_DISTAL"]},
  "geometry_mapping": {
    "type": "LBS",
    "segment_bind_poses": [[0,0,0,0,0,0,1],[0.02,0.01,0,0,0,0,1]],
    "taxel_bind_poses": [[0.03,0.01,0,0,0,0,1]],
    "taxel_segment_indices": [[1,1,1,1]],
    "taxel_segment_weights": [[1,0,0,0]]
  },
  "editor": {
    "version": 1,
    "side": "left",
    "model": {"path": "glove.urdf", "sides": "left", "packages": {}, "asset_hashes": {}},
    "bind_configuration": {"index_joint": 0},
    "wrist_mount": {"link": "palm", "pose": [0,0,0,0,0,0,1]},
    "segment_ids": ["4144c9e8-7b2e-4fe0-8cb0-a29c7336bb61", "98cf8a63-b629-4c09-adc5-ed8c1782a445"],
    "segment_links": ["palm", "index"],
    "taxel_ids": ["2e047330-5224-4f62-ac7a-aae1e5332ebf"],
    "taxel_hints": [{"surface_attachment": null, "provenance": "imported", "input_ref": null}],
    "application_binding": null
  }
}
```

纯数值导出省略 editor。完整文件的数值数组是唯一权威；editor 只保存继续编辑必需的标识、URDF、映射和腕根定义，不再保存另一套 metadata 位姿/权重。model.link_adjustments 保存 4.4 定义的模型局部调整。segment_links 与数组同行；加载时用 B_j 和 q0 恢复局部修正。驱动配置遵循 4.12，不包含相机、选择、预览角度或 live 帧。

不完整或语义非法的草稿保存在同一种 JSON 容器中：editor.version=1、editor.draft 为该侧规范领域记录、原始 skinning 行及模型配置，顶层 segment_source 和 geometry_mapping 省略。draft 与完整数值字段互斥，避免双重来源。此文件可继续编辑，明确不能作为有效 TactileMetadata 交付；正式导出不允许 draft。双手文件各保存对应侧记录，共享模型配置必须一致，否则打开时报告冲突。

导入无 editor 的完整文件时，segment 保持 root 引用，taxel 保持原根位姿及行顺序；生成新 UUID。完成腕根、q0 与链接关联后，使用 A_j=FK_link(q0)^-1*G(q0)*B_j 恢复附件，B_j 和 P_bind 不变。不能要求额外区域或扫描参数，也不能以最近网格点替代原始位置。

保存为 UTF-8，拒绝非有限 JSON 数值；使用同目录临时文件和 os.replace。资源路径相对于 metadata JSON，另存时重算路径以保持实际资源不变。无模型数值文件仍能纯数值检查和导出。合法原始四槽必须逐槽往返；不得因权重等价而改写数组。

### 4.3 事务与状态

修改只通过 Command。数值确认、taxel/segment 关节/URDF 链接拖动结束、批量添加、重新投影、重排及接收分组替换各是一个原子事务；同一次拖动只记录一次。Ctrl+Z 撤销，Ctrl+Shift+Z 重做，macOS 同时支持 Cmd；快捷键由编辑器统一处理，避免与控件本地撤销重复执行。

撤销重做采用本目录内的最小浏览器事件桥接实现。Viser 1.1.1 的标准滑块向 Python 发送数值更新，但其拖动结束只更新浏览器内部状态，内置数值框也使用相同的数值更新通道。因此 ui.py 在现有 Viser 页面中加载 ui_bridge.js，补充操作来源和事务边界；这是既定实现内容，不另建前端或修改外部 Viser 安装。

ui.py 使用标准库 http.server 在 127.0.0.1 临时端口接收桥接事件，浏览器使用 fetch 按序发送 JSON。端口及仅授予当前编辑客户端的会话 token 不持久化、不进入日志或导出；接收端校验 token、允许的 Viser 页面 Origin 和当前编辑客户端身份。预检响应只允许该 Origin，其他客户端只读。接收回调仅投递事件队列，由状态循环处理，不从 HTTP 线程直接修改 Document。

事件携带控件标识、操作来源、gesture_id、顺序号、阶段及当前字段值。ui.py 显式注册参与编辑的控件，按指定 Viser 版本核验其浏览器标识；不靠更新频率、界面文字或延时推测操作类型。桥接行为为：

- 滑块 begin 固定 base_revision、目标设备和稳定实体 ID；update 仅计算临时 Candidate 并预览。end 携带最终值，经 state.dispatch 提交一次；指针移出控件后释放仍须收到结束事件。桥接按手势顺序传输，重复或过期消息不能重复提交。
- 滑块轨道、滑块内置数值框及独立数值框分别标记来源。只有明确滑块操作执行 4.7 的剩余权重分配；直接数值输入在 Enter 或失焦确认时仅提交该字段，连续输入及重复确认不额外生成事务。
- Escape、pointercancel、窗口失去焦点或客户端断开取消未完成的手势，丢弃临时候选，不写撤销历史。不能用防抖超时替代拖动结束；手势期间的长时间停顿不拆分事务。
- Ctrl/Cmd+Z 和 Ctrl/Cmd+Shift+Z 经桥接分别投递 undo/redo，阻止浏览器和输入控件再次执行本地撤销。收到快捷键时先取消未确认的输入或手势，再执行已有文档历史；不同时注册另一条会重复触发的快捷键处理路径。
- 已具有明确开始/结束回调的三维控制器直接接入同一事务入口。对桥接管理的表单，普通 Viser 数值回调不再独立提交 Command，避免数值通道与桥接通道竞争或重复写入。

保留最近 10 次编辑事务及对应重做记录，所有模式可用；撤销后提交新编辑清空重做分支。无限保留文档快照会使长会话内存持续增长，因此采用有界历史，历史不包含网格缓存或实时帧。错误编辑和部分分组替换同样进入历史；引用或数量错误不成为拒绝编辑的理由。

revision 单调递增；dirty 比较规范内容 hash 和最近保存 hash，撤销回保存内容可清除脏标记。候选应用和检查展示核对当前 revision；过期结果不能作为当前通过。导出固定使用启动时快照并标注该 revision，期间继续编辑不改变导出内容。双手共享模型或 q0 改变时，两侧检查都过期。

编辑模式、活动手侧、选中实体、相机、临时批量参数、q_preview、连接、冻结快照属于 ViewState。实时接收不修改 bind_configuration；用户可以在任何模式显式编辑 q0、位姿和权重，这些编辑正常进入历史。回 EDIT 仅恢复 q0 显示；模式切换不自动接受候选。PAUSED 冻结实时三维层，草稿表单、操纵器及撤销历史仍反映当前编辑。

### 4.4 URDF 与局部几何

URDF 单位为米/弧度，支持 fixed、revolute、continuous、prismatic、mimic；planar、floating 和未知类型明确报错。支持基础几何及 STL/OBJ visual，不自动以 collision 替代缺失 visual。零配置作为起点，若超限则提示用户设置合法 q0。

package://pkg/path 通过用户明确的本地目录映射解析，相对路径以 URDF 为基准，不调用 ROS 或网络。由 yourdfpy 建立图，将 visual origin、实例变换和 mesh scale 烘焙到所属链接局部顶点；之后只施加链接刚性 FK。显示和求交使用同一几何，不重复缩放。

非均匀缩放后重算法线，反射网格修正绕序，不把反射写成四元数。geometry_key 由链接、visual 序号和实例路径构成，资源内容 hash 改变即使附件失效；重投影前不能继续使用旧三角形编号。工作线程读取独立 FK/几何快照。

移动 URDF 链接记录为模型配置中的 link_adjustments，不改写输入 URDF。操纵器输入从手根转回父链接坐标，形成固定局部修正，施加于内存模型的父子变换，再由 yourdfpy 求 FK；子树、显示和求交使用同一结果。一次移动可撤销，保持现有 B_j/P_bind 数值并按 4.5 重算附件，不隐式重绑 taxel。

### 4.5 手部来源与绑定姿态

profile.side 只管理应用关联，不进入符号和叉乘规则。HandJoint 候选来自 hand.fbs，排除 NUM_JOINTS；只在 HandProfile 将名称转换为 schema 枚举。segments 列表决定输出顺序，不能把枚举 ordinal 当列表行号。

用户在 q0 定义 O、F、D，构造：

```text
z = normalize(F - O)
y = normalize((D - O) - z * dot(z, D - O))
x = cross(y, z)
T_M_W = pose([x y z], O)
A_w = inverse(FK_wrist_link(q0)) * T_M_W
G(q) = FK_wrist_link(q) * A_w
```

参考投影长度小于 1e-6 m 时拒绝。关节 +Z 朝近端，+Y 朝手背，用户可通过控制器调整，不能假设 URDF 链接轴已符合 OpenXR。对关联链接的 segment：

```text
A_j = inverse(FK_link(q0)) * T_M_J
B_j = inverse(G(q0)) * FK_link(q0) * A_j
C_model_j(q) = inverse(G(q)) * FK_link(q) * A_j
```

合法 WRIST 根位姿应为 identity；允许草稿保留不合法值并显示错误，不静默归零。未关联的导入段以原 B_j 参与静态数组准备；模型运动预览需要所用段已关联，实时 LBS 使用有效 HandPose 当前段位姿，不能据此驱动 URDF。

修改 q0 或腕根时，按新配置保留现有根中的 B_j、P_bind 与权重，再重算已关联 A_j；因此仅设置参考姿态不会偷偷改变 metadata 数组。若用户要改变实际 B_j，则在 Segments 页移动其参考系后提交；这会改变绑定定义并使报告过期。模型不一致保留为待检查现象。

### 4.6 批量几何与范围选择

BatchSpec 仅存于生成候选。所有生成和绑定数据使用 q0 下的手根 W 坐标。anchor_link 的局部拾取先经 T_W_anchor=inverse(G(q0))*FK_anchor(q0) 转到 W：点使用完整位姿，法线和方向仅用旋转。O、C、R、n0、命中位置及后续 T/N 均在 W；查询各链接局部网格时转换射线，命中后立即变回 W。把 C-O、R-O 投影到 O 切平面得到单位列方向 u、行方向 v，跨度除以 cols-1、rows-1 得到间距；单单元维度不做该除法，多单元跨度小于 1e-6 m 报错。

u、v 可形成斜网格，界面显示夹角；abs(dot(u,v))>1-1e-6 时拒绝。单行从列方向求 v=n0×u，单列从行方向求 u=v×n0；单点使用指定切线。修改跨度数值沿已拾取方向改变范围，没有另一个“按间距”模式。

```text
s[r,c] = O + c * column_pitch * u + r * row_pitch * v
ray_start = s[r,c] + depth * n0
ray_direction = -n0
```

depth 默认 5 mm，查询最长 2*depth；只查询指定链接和种子面连通分量，过滤与参考法线夹角大于 75 度的面。可显式翻转所选表面法线。合法命中取距 s 最近者，等距歧义报错，不回退到全模型最近点。局部射线方向仅用旋转转换。

命中使用重心插值法线 N；退化插值可用非退化面法线。切线参考 U 默认为列方向，用户可绕法线旋转 theta：

```text
T0 = normalize(U - N * dot(N,U))
T = cos(theta) * T0 + sin(theta) * cross(N,T0)
Y = cross(N,T)
P_bind = pose([T Y N], hit_position_W)
```

若在链接局部构造 T_anchor_X，等价公式为 P_bind=inverse(G(q0))*FK_anchor(q0)*T_anchor_X；不能将链接局部位姿直接当成模型或手根位姿。

N 先归一化，切线投影长度小于 1e-6 报错。旋转必须正交且 determinant=+1。只处理自身计算的舍入误差，不修复导入坏四元数。投影失败保留空位姿和问题代码；预览种子标记不能冒充 sensing location。排除的单元不生成记录。

首次索引按扫描遍历的未排除候选分配，包含投影失败的空位姿记录，不能因失败而压缩索引。row-major 行外列内，column-major 相反；snake 在外层奇数遍历反转内层。reverse_rows/reverse_cols 先改变轴顺序再 snake。默认起始值寻找足以容纳本批次且无冲突的连续区间；应用后不保留行列身份。

框选将当前视图中的 taxel 中心投影到屏幕，与矩形比较，过滤相机后方、其他手侧和遮挡点。使用 Viser on_rect_select 的 screen_min/screen_max；选择结果基于手势完成时固定的相机和场景快照。穿透选择省略网格遮挡检查。索引范围选择不依赖相机。

### 4.7 权重与 LBS

编辑器保留原始四槽的索引、权重及顺序；相同 segment 出现在多个槽中也不合并。例如 [j,j,j,j] 与 [0.25,0.25,0.25,0.25] 导入、显示、保存及导出后仍为这些值。

此处“校验器”特指仓库函数 [core::verify_tactile_metadata()](../../src/core/schema/cpp/inc/schema/tactile_metadata_validation.hpp)，由 SDK 的 TactileMetadata 构造函数调用。joint_ids 定义每个 segment 是哪个语义关节；taxel_segment_indices 存的是对该列表的行号引用。例如：

```text
joint_ids = [WRIST, INDEX_DISTAL]
taxel_segment_indices[0] = [1, 1, 1, 1]
taxel_segment_weights[0] = [0.25, 0.25, 0.25, 0.25]
```

这里仅定义一次 INDEX_DISTAL，四槽共同引用它所在的第 1 行，允许重复引用。被拒绝的是 joint_ids=[WRIST, WRIST]，因为它将同一语义关节重复定义为不同 segment 行。定义重复与引用重复是不同检查；该行号 1 也不是 INDEX_DISTAL 的枚举值 9。

滑块显式修改第 k 项为 x 后，其余参与项按 w'_i=(1-x)*w_i/sum_remaining 分配；sum_remaining=0 时均分，仅一个参与项则固定为 1。直接输入数值和接收值替换不执行这条规则。操作来源及一次拖动的事务边界由 4.3 的事件桥接确定；多选混合值不改变原值。

只有新建 taxel 的初始化才为未用槽填入合法 segment 索引和零权重，例如新建单段绑定可用 [j,j,j,j] 和 [1,0,0,0]。初始化不适用于已有或接收的槽。检查逐槽报告缺项、越界、负权重及非法和；错误值保留，不补齐、不截断、不归一化。导出只读取当前草稿中的槽值，所有零权重槽同样需要合法索引。

位置预览按每个原始槽计算；只对变换后的点求加权和，不汇总或改写权重。静态/运动预览的 C_j 来自模型 FK，实时 C_j 固定来自 HandPose，没有来源选择控件：

```text
p_W(t) = sum_k w[k] * C[j[k]](t) * inverse(B[j[k]]) * p_bind
p_view(t) = V_side(t) * p_W(t)
```

编辑/URDF 预览中 V_side=G(q)，实时腕根视图中 V_side 为该侧显示变换；两者不混用。单段方向用 C_j*inverse(B_j)*P_bind；多段不求唯一刚性方向。力模长不依赖此方向，可始终用于有效对应点的颜色。

### 4.8 数组准备、输出顺序与 dtype

compile_device 的含义仅是准备供检查、预览和导出使用的协议数组，不是修改编辑数据的过程。有效 metadata 按 taxel_index 排序，必须恰好覆盖 0..N-1。空设备及索引空缺可作为错误草稿编辑，但不能正式导出；不宣称 schema 本身拒绝所有空向量。删除导致的空缺需要用户显式重排，不把点移动、重投影或实时输入顺序当作输出重排。

segment 顺序来自 segments 列表，HandProfile 生成 joint_ids；例如 [WRIST,INDEX_DISTAL] 对应枚举 [1,9]，第二段影响索引为 1。全部 B_j 和 P_bind 在同一个腕根中，四元数 XYZW、位姿宽 7、影响宽 4。

原始索引和权重按编辑器当前值逐槽读取，不做关节去重、槽排序、权重累加或自动归一化。计算使用 float64，转换前检查整数性、范围及有限值；末尾生成拥有内存的 C-contiguous float32 位姿/权重和 uint32 索引。转换后再次检查溢出与精度，不修改草稿，不能先强制转换负数/小数索引后验证。

### 4.9 本地检查与协议事实

先检查模型、设备、引用、资源和输出索引，再准备数组；失败 compiled=null，保留定位信息。数组检查包括形状、joint_ids 唯一性、所有槽索引、有限位姿、单位四元数、非负权重及权重和；不将重复 skinning 索引误报为重复 joint_ids。错误直接标在所属分组和 Info 区，不回滚错误编辑或静默替换原值。

共享校验器要求权重和误差不超过 1e-5，四元数平方范数误差不超过 1e-4。权重按已转换 float32 的值用双精度累加，范数与原生 float32 对齐；最终官方结论来自 SDK 构造。

local_status 为 passed/failed；sdk_status 为 not_run/unavailable/failed/passed。缺字段、缺绑定、输出索引冲突和数值错误是 error；几何估计、运动近似与无法自动证明的物理真实性是 warning。无 URDF 的完整数值 JSON 不因缺可视化附件而无法数值导出；引用了失效附件时必须清理或重建。

依据：[tactile_metadata.fbs](../../src/core/schema/fbs/tactile_metadata.fbs)、[hand.fbs](../../src/core/schema/fbs/hand.fbs)、[tactile_array.fbs](../../src/core/schema/fbs/tactile_array.fbs)、[共享校验器](../../src/core/schema/cpp/inc/schema/tactile_metadata_validation.hpp)。

### 4.10 SDK 调用：实时接收与导出编码

实时接收必须实际调用 Isaac Capture SDK 的 tracker 和 session 类。GUI 通过本工具的本地 live_worker 使用 SDK，worker 持有 tracker/session 实例并返回只读数据快照；SDK 使用不限于导出时的序列化。

| SDK 类及模块 | 实际用途与调用 |
| --- | --- |
| isaaccapture.deviceio_trackers.TactileTracker | 按 collection_id 构造；get_metadata(session) 读取 TactileMetadata，get_data(session) 读取 TactileDataArray。 |
| isaaccapture.deviceio_trackers.HandTracker | 构造共享实例；get_left_hand(session)/get_right_hand(session) 读取配置手侧的 HandPose。 |
| isaaccapture.deviceio_trackers.JointStateTracker | 启用 JointState 时按 collection_id 构造；get_data(session) 读取 JointStateOutput，用于具名 URDF 驱动。 |
| isaaccapture.deviceio_session.DeviceIOSession | get_required_extensions(trackers) 获取所需扩展；run(trackers, handles) 建立接收会话，session.update() 更新后再读取各 tracker。 |
| isaaccapture.oxr.OpenXRSession | 使用所需扩展建立本机 OpenXR 会话；get_handles() 将句柄交给 DeviceIOSession.run。 |
| isaaccapture.schema.HandSkeletonRef、LBS、TactileMetadata | SDK 检查及二进制导出时构造元数据并调用 to_bytes()；不承担 tracker 的实时接收工作。 |

实际接收顺序沿用[直接消费示例](../../examples/tactile/python/isaaccapture_examples/tactile/printer.py)：构造配置的 tracker，聚合 required extensions，进入 OpenXRSession 和 DeviceIOSession.run，在循环中调用 session.update()，再读取 metadata、触觉力、HandPose 和可选 JointState。返回值及 validity 的复制、快照传输和退出流程见 4.11。没有 runtime 或设备时显示真实连接错误；合成演示不替代这条 SDK 接收路径。

live_worker 是长生命周期接收进程；sdk_worker 是环境探测、官方构造校验及编码的短进程。二者均通过参数数组和 `sys.executable` 在本地启动，继承调用者的环境和模块搜索配置，设置已有 ISAACCAPTURE_LOGGING=off。不经 shell、不选择其他解释器、不注入固定 SDK 路径，也不建立或安装另一份 SDK。SDK 在启动环境不可用时显示原因，离线编辑和本地 JSON 检查仍可运行。GUI 主进程不直接导入 isaaccapture，避免其[初始化](../../src/python/isaaccapture/__init__.py)加载多个原生子模块。

sdk.py 的短进程调用超时为 30 秒。临时请求包含 version:1、operation:probe/encode、values_file、report_file、binary_file。报告包含 status、SDK 版本、支持类型、关节名称/值、metadata_max_size、encoded_size 和 issues。stdout/stderr 只作诊断，结果通过文件传递；实时长进程使用 4.11 的快照协议。

encode 验证 dtype 范围后调用[公开绑定](../../src/core/schema/python/tactile_metadata_bindings.h)：

```python
hand = HandSkeletonRef([getattr(HandJoint, name) for name in joint_names])
lbs = LBS(segment_bind_poses, taxel_bind_poses, taxel_segment_indices, taxel_segment_weights)
metadata = TactileMetadata(geometry_mapping=lbs, segment_source=hand)
encoded = metadata.to_bytes()
```

与该 SDK 的 TRACKER_METADATA_MAX_SIZE 比较，不硬编码上限。复制回读数组核对顺序和值，现有[数组视图](../../src/core/schema/python/schema_array_views.h)可写，不得作为编辑状态。父进程核对报告、退出码、实际长度，读取 bytes 后才清理临时目录。导入/超时为 unavailable，类型/语义/大小错误为 failed。

[序列化绑定](../../src/core/schema/python/schema_serialized.h)只有 to_bytes，没有 from_bytes；不实现虚构的裸二进制导入。

### 4.11 实时接收与生命周期

live_worker 执行 4.10 的 SDK 调用：每个触觉 collection 一个 TactileTracker，每个启用的具名运动来源一个 JointStateTracker，HandTracker 共用实例读取左右。左右 HandPose 使用该实例共同的 vendor 配置。构造参数与支持类型依据公开绑定及[tracker manifest](../../src/core/deviceio_trackers/trackers.toml)；不自动枚举来源，不假设相同 collection 命名就完成关联。

配置 VendorConfig 的 vendor id/params 时，扩展查询和 session 创建使用相同配置，不新增私有供应商协议。SDK/runtime 不可用时返回可定位错误，离线文档继续存在。

读取 tracker 返回值后，HandPose 的位置、XYZW、radii 和 is_valid 通过[公开批量视图](../../src/core/schema/python/hand_bindings.h)复制，JointState 依[公开字段](../../src/core/schema/python/joint_state_bindings.h)按名称解析，触觉 metadata 和力值同样复制到工具自有数据。所有数组在下一次 update 前复制，不跨线程保留 SDK buffer。

父进程在 127.0.0.1 临时端口监听，把端口、会话 token 和配置写入仅当前用户可读的临时请求文件。worker 只连接该本机端口，握手校验 version/token；数据为 4 字节网络序长度加 UTF-8 JSON，单消息上限 8 MiB，禁止 NaN/Infinity。stdout/stderr 不作协议。控制消息为 stop，数据消息为 status、metadata、frame、error；token 不进入日志或导出。

metadata 消息包含规范数值 JSON 和 SHA256 指纹：排除 editor，关节用名称，数组先转目标 dtype，再用固定键序、紧凑分隔和有限数值序列化；保留原始四槽顺序及权重，不累加或归一化。相同比较使用同一路径。frame 包含 epoch、seq、read_monotonic_ns、各来源的值及有效性。worker 每次 update 读取现有值，发送频率上限 30 Hz；GUI 只保留最新完整快照，不积压历史。配置来源变更需停止并重建本工具接收 session。disconnect 发送 stop 到本地 worker，等待最多 5 秒，再终止并回收子进程。

协议的时间是本机读取时间。当前 tracker getter 不提供统一的采集时间，不得把同次 update 称为硬件同步，也不能把缓存数据重新读到就标成“生产者新样本”。HandJoint/JointState 的 validity 必须保留；worker 无心跳 1 秒标连接中断，值不可用时灰显。生产者缓存是否停更不能仅靠读取时间判定，页面明确时间含义。

本工具仅是接收端，不持有生产端可写 buffer，不发布 metadata 或控制命令。此处的错误指本地 SDK 接收失败，例如收到与已采用基线不同的 metadata、无效 buffer、runtime 异常或 worker 退出，不是编辑草稿影响了发送端。按[接收契约](../../docs/source/device/tactile/index.rst)，致命异常结束本工具的接收 worker 并显示原因，GUI 和草稿继续可用；共享 session 中的其他输入也标记为接收中断。不要求发送端停止或接受编辑值。普通缺帧或 validity=false 只影响相应数据，不拿另一侧补齐。

### 4.12 实时映射、源索引与编辑结合

application_binding 仅保存每侧 tactile_collection_id、hand_source、可空 joint_source、具名 JointState 映射和 source_taxel_map。验证 hand_source.side 与当前设备相同；触觉 metadata 本身没有侧别，只能依用户声明绑定，不能宣称自动验证了真实设备手侧。端口和当前帧不持久化。

记录 metadata 是从接收到草稿的单向替换，只写入勾选分组：Skeleton binding poses 包含 joint_ids 和 segment_bind_poses，Taxel binding poses 包含 taxel_bind_poses，Skinning 包含 taxel_segment_indices 和 taxel_segment_weights。默认全选，每组内部字段不可拆选；未选组保持原值。先成功备份当前草稿，再经 Candidate/state 在一次可撤销事务中替换；备份失败则保留原草稿并显示文件错误。接收帧和切换三维来源不执行替换。

按接收数组的原行号、原槽顺序及原数值替换，不与旧值求和，不根据名称、数量或最近几何自动匹配。部分替换后允许行数、索引或引用不一致，保留原始数据并在 Skeleton、Taxel 或 Skinning 对应区域及 Info 区显示原因；不要求先修复才能继续编辑或撤销。替换 skeleton/taxel 组时为替换后的记录建立编辑 ID，不沿用旧表面附件或链接修正；这些编辑信息可重新关联，数值不得因此改变。

具名映射每行包含 urdf_joint、source_joint、scale、offset，q=scale*position+offset；检查 source device_id、名称唯一性、valid 和单位，禁止按数组下标匹配。未映射自由度用 q0；mimic 由 URDF 处理。限制越界仅用于显示，保留原值并报告，不能修改输入样本。

启用 joint_source 时，实时 URDF 仅使用具名 JointState 映射；未启用时隐藏。不提供 HandPose→URDF 映射、手关节 twist 提取或中立旋转校准，也不以 HandPose 补齐缺失或无效的 JointState。

实时 LBS 固定使用 HandPose，C_j=inverse(H_WRIST)*H_joint。所需腕/关节无效时相应 taxel 标记不可用，不回退或混合模型 FK。模型 FK 仅用于 q0 编辑及手动运动预览，不提供 LBS 来源切换。

实时视图统一使用手根空间：骨架为 inverse(H_WRIST)*H_joint，启用的 URDF 为 inverse(G(q))*FK_link(q)，taxel 使用同根 LBS 位置。双手可施加仅用于画面排布的固定偏移 V_side，所有图形一致使用它，但它不进入 metadata。无需跨设备世界坐标的外参变换或额外根位姿来源。

源对应每项为 taxel_id→input_taxel_index，并绑定 tactile collection、手侧及 metadata 指纹。替换 Taxel binding poses 时按接收原行建立；只替换 skeleton 或 skinning 不新建 taxel 对应。单纯修改几何/权重保留对应，输出重排不改变，新增点无对应。独立 JSON 数值与基线完全一致可自动建立，否则需用户显式建立表格对应；数量相等不够。基线指纹改变则整个对应过期；越界或多个 taxel 使用同一源行均报错。

对应有效才着色，数据长度必须等于生产端 metadata 的数量；当前编辑器 N 可以不同。使用旧校准力显示新轴向时提示这是布局检查，不是重新标定。冻结快照拥有数组内存；返回编辑只改 bind 数据，恢复播放取最新帧并重新计算，不把冻结角度、力值或 C_j 写入 B_j。

### 4.13 导出与可视化实现

对同一 Snapshot 进行检查与编码，先写同级临时目录，全部选定手侧成功后发布新目录。非空目标拒绝覆盖。每侧生成 <side>.metadata.json（含 editor）、<side>.values.json（纯数值）、<side>.taxels.csv 和可选 <side>.metadata.bin；另有 manifest.json、report.json。

taxels.csv 列为 taxel_index,taxel_id,input_taxel_index，按输出索引排序，没有区域和硬件电气标识。manifest 的 format=isaaccapture.tactile-package、version=1，包含 revision、device_id、side、文件名/hash 和可空 collection_id/hand_stream_ref；非空 collection_id 不重复。report 保存 warnings、检查状态及实际字节数。所有产物仅写本地文件；本工具不将导出内容发布到 runtime，manifest/editor 也不是新增运行时协议。

UI 使用标准表单与页签，浏览器事件桥接和事务提交遵循 4.3，回调只投递队列。taxel 批量绘制球体及线段，仅选中项创建控制器。点击通过射线与球解析求交，不依赖批量节点 instance_index；框选见 4.6。局部加速结构缓存，运动只更新变换，显示更新上限 30 Hz，不以配置值声称已达到性能。

CLI 使用 argparse，成功 0、数据或检查失败 1、参数/格式/版本错误 2、请求 SDK 但不可用 3。日志为 isaaccapture.tools.tactile_metadata.<module>；独立 GUI 不导入 SDK，可按仓库独立程序例外配置日志，诊断进日志，CLI 帮助属于终端 UX。__init__.py 不做 SDK 或模型加载。

## 5 给 agent 的范围规定风格规范和 restriction

### 5.1 范围与工作规范

实现第一至第四类定义的完整工具，工具代码及模型示例位于 scripts/tactile_metadata/。允许为本工具补齐 Isaac Capture 现有环境依赖声明及对应包元数据；不建立工具专用环境。不改 schema、golden 或无关子系统；真实冲突提供代码证据并同步规格。

修改前列出目录并读取全部适用 AGENTS.md，保护已有改动。本地检查，不连接远程计算机，不自动提交或推送。不把本机 SDK/构建/资源绝对路径硬编码到工具或可移植文件。

领域类型集中，GUI/CLI 共享数组准备、检查和导出；按模块契约实现，不另造通用框架。遵守仓库日志、SPDX、简短注释、格式化和 Mandatory learning loop。文件输入范围、术语和批量操作生命周期是产品约束，不能为实现便利恢复被移除入口。

### 5.2 实施顺序与交付

先接通 domain/document/state，再实现 URDF、Segments、批量几何和原始四槽编辑；接通数组准备、检查、SDK 和导出后，实现 live/driver 及六页 GUI。每阶段核对数据产物，最后完成离线与实时完整路径。

交付复用启动 Python 环境的工具、Isaac Capture 环境依赖声明更新、基于本目录现有 model/ 中 URDF 的 metadata JSON 示例、README、开发者技术说明和可验证的导出包。不交付工具专用环境或锁文件，不要求新增 URDF、网格或测试文件，也不新增测试依赖。合成实时源可在无 SDK/硬件时运行，实际 live 接收仍须实现，不能用合成数据代替。README 逐步说明在已有 Isaac Capture 环境中准备依赖并启动工具，以及离线编辑、JSON 继续编辑、原始四槽、源行对应和实时播放/暂停/编辑/恢复；技术说明解释模块和数据流。

### 5.3 必须执行的验证

| 验证 | 独立预期结果 |
| --- | --- |
| 批量范围 | O=(0,0,0)、C=(0.009,0,0)、R=(0,0.006,0)，3×4：间距均 3 mm，row-major 索引 3 为 (0.009,0,0)，4 为 (0,0.003,0)。 |
| 四槽及语义 | [WRIST,INDEX_DISTAL] 枚举为 [1,9]，第二段行号为 1；重排保持物理引用，所有零权重槽仍合法。 |
| 原始槽往返 | 同一段的 [0.25,0.25,0.25,0.25] 逐槽保留，不变为 [1,0,0,0]；joint_ids 重复报错，不将 skinning 的重复索引误判为该错误。 |
| 坐标 | N=(0,0,1)、T=(1,0,0) 得 Y=(0,1,0)，行列式为 +1，两手一致。 |
| 手根转换 | G(q0) 平移 (0.1,0,0)，FK_anchor(q0) 平移 (0.12,0.01,0)，旋转均为 identity；局部点 (0.03,0,0) 转为手根点 (0.05,0.01,0)。 |
| 权重界面 | (0.2,0.3,0.5) 第一项改为 0.6，得到 (0.6,0.15,0.25)；高亮关节和所选点；导入坏权重不自动修正。 |
| LBS 数值 | B1 原点 (0.02,0.01,0)，C1 同原点绕 Z 转 90°，p=(0.03,0.01,0)：单段得 (0.02,0.02,0)，与静止根各 0.5 得 (0.025,0.015,0)。 |
| JSON 往返 | 无 editor 的数组载入 URDF，链接关联前后 B/P 不变；保存重开继续编辑；草稿与完整数值来源互斥；无 Region 依赖。 |
| 手侧 | 导入后 sides 不可改，双手不镜像；左右不同 N/顺序，切换及撤销不串侧，共享 q0 的检查同时过期。 |
| 模型 | 非零 visual origin、非均匀 scale、mimic、旋转平移；显示与求交一致，非允许表面不能命中。 |
| 实时映射 | 具名值乱序仍匹配，错误 device_id/无效值拒用；未启用 JointState 时实时 URDF 隐藏，HandPose 变化不驱动 URDF；非零 base 下 wrist rebasing 正确。 |
| Metadata 分组 | 默认全选 Skeleton binding poses、Taxel binding poses、Skinning，字段分组与 1.8 一致；显式记录先备份草稿并显示路径，再在一次事务中替换所选组，未选组不变；允许替换后错误，显示具体原因；切换三维来源不替换表单。 |
| 撤销重做 | 连续 10 次数值修改、taxel/关节/URDF 链接移动及接收分组替换可经 Ctrl+Z 撤销、Ctrl+Shift+Z 重做；模式不限制操作，恢复数值、引用和错误状态。 |
| 浏览器事件桥接 | 连续拖动及手势内长时间停顿只产生一个事务，控件外释放仍提交；取消不写历史，重复消息不重复提交。直接输入不重新分配其余权重，Enter/失焦不重复提交；Ctrl/Cmd 快捷键只执行一次文档撤销或重做。 |
| 实时隔离 | 所有表单始终显示可编辑草稿；暂停继续接收但实时三维层冻结，草稿操纵器仍响应编辑；恢复用最新帧和当前草稿；接收或播放本身不修改草稿，任何编辑均不发布到 runtime。 |
| 源行对应 | 同数量不同基线不自动匹配；重排只改变输出索引；新增点无力值；metadata/epoch 改变不得沿用过期数据。 |
| 失败路径 | 缺位置/方向、坏四元数、NaN、非法/缺失索引和资源变化显示定位原因，草稿仍可编辑；未知版本、SDK 超时、worker 崩溃及接收异常可见；致命接收错误不影响草稿或发送端，双侧导出失败无部分文件包。 |
| 启动环境复用 | GUI/CLI 使用调用者选定的 Python；SDK/live worker 使用同一 sys.executable 并继承环境。无 SDK 时离线流程仍可运行，不查找旧 SDK、创建环境或触发构建。 |
| 官方 SDK | 启动环境具备 SDK 时，验证实际构造、数组回读、to_bytes、大小上限；真实 tracker API 与会话错误可见。条件不具备时标注未验证，不能以 mock 宣称官方通过。 |

使用 model/ 中现有 URDF 完成浏览器路径：载入模型 → Segments/q0 → 批量添加 → 范围多选及改权重 → 保存重开 → 合成实时播放/暂停/编辑/撤销/恢复 → 检查并导出核对。启动环境具备 SDK/runtime/硬件时，再验证实际 SDK 接收及二进制导出；条件不具备时明确列出未验证项，不为验收另建环境或构建 SDK。双手路径只使用实际包含双手的可用 URDF，不为验收额外创建模型；真实接收实现及合成全链路仍须交付。

验证通过临时数值计算、实际文件往返、SDK 构造和浏览器操作完成，不新增测试文件。基于现有模型记录 taxel/三角形数量、生成、框选和运动更新耗时、CPU/内存及环境；约 1000 taxel、100000 三角形作为规模参考，不要求新增模型资产。文档入口修改先读 docs/AGENTS.md 并执行要求。

检查仅从 scripts/tactile_metadata/ 或其子目录执行，并限定本任务文件；不从仓库根或 /tmp 运行 pre-commit 或测试，也不运行可能改写其他目录的全仓库自动修复钩子。修复可执行检查发现的问题，按 Mandatory learning loop 记录经验；受范围或环境限制未执行的检查明确列出。最终报告实际功能、启动命令、产物核对与未验证项；离线编辑、实时接入、GUI、检查及导出接通后才报告实现完成。
