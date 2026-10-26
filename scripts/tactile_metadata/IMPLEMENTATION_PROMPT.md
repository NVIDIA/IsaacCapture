<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Tactile Metadata Editor 实现提示词

将以下提示词交给能够读取本仓库的实现代理。完整规格见同目录 [DESIGN.md](DESIGN.md)。

```text

请实现 /Users/lotusl/Codes/IsaacTeleop/scripts/tactile_metadata/DESIGN.md
定义的 Tactile Metadata Editor。

When implementation, make sure to only modify folder /Users/lotusl/Codes/IsaacTeleop/scripts/tactile_metadata and its decendent files. It's where your implementation should fully locate. Don't write to any other paths, except for /tmp and your own workspace, and the IsaacTeleop folder's library config only to add the library dependencies as guided. Don't connect to any remote computer or write to any remote end. You're allowed to read codes in the whole ~/Codes/IsaacTeleop. Don't run pre-commit or other whole-repo formatter/tests. It's OK to run local formatter/test within tactile_metadata.

先完整读取设计、仓库根及所有修改路径适用的 AGENTS.md，检查已有改动。
这是实现任务，按已确定的规格工作。编码前完整了解领域、数据流及已有
能力，以现有 schema、公开绑定、共享校验函数和 tracker/session 核实
协议。若发现尚未解决的规格矛盾、错误假设、风险、缺失功能或 SDK 能力
缺口，先暂停并提供具体证据，请用户决定。以下内容均为已确定的要求。
Also analyze whether any operation would require my manual permission. If yes, pop these permission requirements first.

可在既定范围内通过局部实现补齐的库回调不属于待决规格；下述浏览器
事件桥接已确定，直接实现，不为此另行请求许可或改变用户交互。

按以下五类实施。

1. 用户界面和用户功能设计

实现模型、Segments、布局、绑定、实时、导出六页及共享三维视图。
只支持 URDF 和 TactileMetadata JSON 文件输入；STL/OBJ 仅可作为 URDF
资源，不提供独立网格、CSV 或裸二进制导入，也不另建工程文件格式。
JSON 的 editor 扩展用于继续编辑，无扩展的数值文件同样必须可用。

载入 URDF 时确定 left/right/both，之后不能改。双手必须来自实际包含
双手的 URDF，不隐式镜像。编辑对象切换只选择 metadata，不改变手侧。
不要求预填 taxel 数量；数量从记录推导，统一使用 taxel 索引和输出顺序。

Segments 页选择 HandJoint、关联链接、调整关节参考系和 q0。导入数组
后关联 URDF 必须保持 B_j/P_bind 数值，不能以链接原点或最近表面替代。
显式重排 segment 保留稳定引用并更新行号，语义枚举值不能当影响行号。

批量添加只有范围加行列数量一种方式。提供单点、框选和索引范围多选，
正确处理单行、单列、单点、排除单元和投影失败。没有区域名或持久化
Region/批次对象；应用后仅留下独立 taxel，导入结果与新建结果等价。

绑定只有一套四槽索引和权重界面，保留原始槽顺序和数值，包括重复
segment 引用；[0.25,0.25,0.25,0.25] 不能改写为 [1,0,0,0]。
只有显式拖动滑块才按设计分配剩余权重；直接数值输入仅修改该字段。
高亮对应关节、链接和所选 taxel；多选混合值不伪造统一值。不恢复独立
刚性、两段过渡或手工模式，不自动累加、归一化或修复已有及导入槽值。

所有模式下草稿始终可编辑，包括 PREVIEW、LIVE、PAUSED 和连接中断。
六页表单始终显示草稿值；接收 metadata／草稿来源切换只影响三维显示。
表单和草稿操纵器编辑 q0 下的手根数据，不反解实时 LBS。暂停只冻结
实时三维层，草稿操纵器仍响应编辑；恢复使用最新帧和当前草稿。

允许错误草稿，包括缺失引用、错误权重、索引越界和数组行数不一致。
保留原值，在字段和所属分组显示错误状态与原因，并在三维视图下方提供
小型 Info 区汇总。错误不阻止编辑、替换、撤销和重做；正式导出仍须合法。

“记录 metadata”是从接收到草稿的单向分组替换，默认全选：
- Skeleton binding poses：joint_ids、segment_bind_poses；
- Taxel binding poses：taxel_bind_poses；
- Skinning：taxel_segment_indices、taxel_segment_weights。
按原行号、原槽顺序和原数值替换所选组，未选组不变。不合并旧值、不
自动匹配语义或修复引用；即使替换后有错误也保留结果并显示原因。
替换前先成功备份草稿到临时位置并在日志显示路径；备份失败保留草稿。
每次替换是一个可撤销事务；接收帧或切换三维来源不自动替换草稿。

实现 Ctrl+Z／Ctrl+Shift+Z 撤销和重做，macOS 同时支持 Cmd。保留最近
至少 10 次编辑事务，包括数值修改、taxel/关节/URDF 链接移动、q0 修改
及接收分组替换；一次拖动只提交一次。不将实时帧或网格缓存存入历史，
用有界历史避免长会话内存持续增长。

按 DESIGN 4.3 实现 ui_bridge.js 与 ui.py 的本地浏览器事件桥接，作为
撤销重做的实现。复用现有 Viser 页面，用标准库 http.server 在
127.0.0.1 临时端口接收带当前编辑客户端 token 的有序事件；校验页面
Origin 和编辑身份，回调仅投递状态队列。不另建前端，不修改外部库。
明确区分滑块与数值输入，发送 begin/update/end/cancel、手势标识和
最终值；拖动只预览候选，结束后提交一次。控件外释放仍结束，取消不
写历史，重复消息不重复提交，不用防抖超时猜测手势结束。
直接输入只在确认时修改该字段，Enter/失焦不重复提交；不得由 Viser
原始数值回调再次提交。Ctrl/Cmd 撤销重做走同一桥接，先取消未确认输入，
再操作文档历史，并阻止控件或浏览器重复执行本地撤销。

实时页实际接通 TactileTracker、HandTracker 和可选 JointStateTracker。
实时 URDF 只由启用的具名 JointState 映射驱动；禁用时隐藏 URDF。
HandPose 只提供实际骨架及实时 LBS 段位姿，不驱动 URDF。取消手关节
twist 映射、中立旋转校准、SE3 输入、LBS 来源选择和世界坐标外参标定。
明确来源到手侧和源 taxel 行的对应，不能只因数量相等就关联。
工具仅接收数据；编辑、保存及导出都不向 runtime 发布 metadata 或
运动控制命令，不修改发送端。

2. 依赖

采用 DESIGN 第 2 类规定的 Python、Viser、yourdfpy、trimesh、NumPy、
rtree 和标准库通信。将缺失库及所需版本约束加入 Isaac Capture 现有
src/core/python/requirements*.txt 的对应分组，并同步根 pyproject.toml
等对应包依赖声明；已有依赖按所需版本复用或补充约束。
工具复用启动它的 Python 环境，以 python -m tactile_metadata_editor.cli
从工具目录运行。不创建工具专用 pyproject.toml、uv.lock、虚拟环境或
另一份 SDK；不使用 uv run --project 为工具切换环境。
核对共享依赖声明及实际安装版本，验证模型加载和指定版本的 Viser
框选 API。不新增测试依赖，工具启动时不自动安装库或触发原生构建。
SDK/live worker 通过 sys.executable 使用与启动工具相同的解释器，继承
调用者环境和模块搜索配置，复用其中可导入的 SDK。不提供 SDK 解释器
或路径配置，不查找旧构建目录，也不注入固定 SDK 路径。
离线编辑/JSON 检查不导入 SDK；SDK 检查、二进制导出和实时接收需要
启动环境中的 SDK，实时接收还需要对应 runtime/设备。不可用时显示原因，
离线编辑仍可运行；不要求实现机器预先具备未来用户的启动环境。

3. 代码架构

按 DESIGN 第 3 类建立 domain/document/state、model/layout、
profiles/hand、mappings/lbs、compiler/checks、sdk/sdk_worker、
live/live_worker、drivers、export、ui/cli 及对应接口。
Document 是唯一编辑数据源；生成返回 Candidate，state 负责原子应用
编辑及分组替换。保留独立的原始 skinning 行，以容纳行数不一致的错误
草稿，不补造 taxel 或另外保存一套汇总权重。
GUI/CLI 共用数组准备、检查和导出。compile_device 仅排列输出行、检查
引用及转换协议 dtype，不编译源码、不按关节聚合或修改草稿。
错误草稿返回可定位的 CompileFailure，CheckResult 保留 compiled=null
和 issues，不以假数组伪装成功，也不因此禁止草稿编辑。

profile/mapping 分别注册，通用领域模型不直接依赖 HandJoint。
SDK 导入仅发生在独立本地子进程：sdk_worker 负责探测、官方校验及编码，
live_worker 长期接收真实数据。进程隔离规定调用位置，不免除 SDK 调用。
live_worker 必须实例化 TactileTracker、HandTracker 和可选
JointStateTracker，获取 required extensions，建立 OpenXRSession 和
DeviceIOSession.run，会话循环 session.update() 后读取 tracker getters。
具体类、模块和调用见 DESIGN 4.10。LiveSnapshot 与 Document 分离，
复制为自有快照，不共享 SDK 可写视图；合成数据不能代替真实接收实现。

4. 每个模块内部的逻辑、实现和 convention

严格实现 DESIGN 第 4 类，包括：
- 米/弧度/XYZW、稳定 ID、null 缺失位姿、Y=normal×tangent；
- metadata JSON/editor 扩展、完整数组与草稿互斥、相对路径和原子保存；
- 单调 revision、内容 hash 脏标记、事务、撤销重做及过期结果处理；
- URDF FK、visual origin/scale、mimic、资源 hash、局部几何缓存及
  文档内 link_adjustments；链接移动不改写输入 URDF 或隐式重绑 taxel；
- q0 手根空间中的 B_j/P_bind；局部生成位姿使用
  P_bind=inverse(G(q0))*FK_anchor(q0)*T_anchor_X，方向只用旋转转换；
- 范围生成、受限射线、方向构造、临时排除及投影失败；失败记录保留
  空位姿和输出索引，不因失败而压缩索引；
- 原始四槽校验及 LBS 位置预览，只对变换后的点求加权和，不改写权重；
- taxel 输出索引、源输入索引、段行号及 dtype 边界；
- 本地校验、SDK 官方构造/回读/实际大小和文件结果协议；
- 本地实时 worker 的握手、长度前缀协议、生命周期及最新快照；
- 具名 JointState 映射及 validity；实时段位姿固定为
  C_j=inverse(H_WRIST)*H_joint，不逐关节回退到模型 FK；
- metadata 指纹保留原始槽顺序和值；源行对应、实时层冻结和恢复；
- 一致快照的全侧导出、批量绘制、框选、CLI 退出码和日志。

协议校验依据是 core::verify_tactile_metadata()，位于
src/core/schema/cpp/inc/schema/tactile_metadata_validation.hpp，由公开
TactileMetadata 构造函数调用。joint_ids=[WRIST,WRIST] 是重复定义，
会被拒绝；joint_ids=[WRIST,INDEX_DISTAL] 下的 [1,1,1,1] 是四槽重复
引用同一个 segment 行，允许存在。不可混淆定义、行号引用和枚举值。

接收错误指本地 SDK/runtime 失败，不是草稿影响发送端。致命异常可结束
本工具的共享接收 worker，但 GUI 和草稿仍可用；普通缺帧或无效样本
只影响对应来源。不得要求发送端停止或接受编辑值。

不虚构 from_bytes、自动设备发现或采集时间接口，不把一次 update
当成多设备硬件同步，不将缓存读取时间称为生产者更新时间。
多关节 LBS 不定义唯一刚性朝向，只显示位置和力模长；不能隐藏几何偏离。
不静默压缩输出索引、镜像手侧、修复导入数据或写入 SDK buffer。

5. 给 agent 的范围规定、风格规范和 restriction

工具代码及基于现有 model/ 中 URDF 的 metadata 示例位于
scripts/tactile_metadata/；允许为本工具补齐 Isaac Capture 现有环境依赖
声明及对应包元数据。不新增测试文件、测试框架、URDF 或网格。
不改 schema、golden 或无关子系统，不建立工具专用环境。
保护用户已有改动，全部本地执行，不连接远程计算机，不自动提交推送。
遵守日志命名、SPDX、简短注释、格式化和 Mandatory learning loop。
检查仅从 scripts/tactile_metadata/ 或其子目录执行，并限定本任务文件；
不从仓库根或 /tmp 运行 pre-commit 或测试，不运行可能改写其他目录的
全仓库自动修复钩子。受范围或环境限制未执行的检查明确列出。

保持工程简洁，只实现明确要求的功能及其必要鼠标/键盘交互。不增加
无关用户功能、通用框架、不必要的工具函数或已有库功能的重复实现。

先接通领域、几何和数组准备，再接 SDK、导出、实时接收和 GUI。通过
临时数值计算、实际 JSON 往返、SDK 构造及浏览器操作执行设计中的验证，
不新增测试文件。核对手侧隔离、错误草稿、原始槽往返、分组替换、
浏览器桥接的拖动/确认/取消及单事务撤销重做、播放/暂停/编辑/恢复，
以及实际 SDK 产物和离线/实时完整路径。
基于现有模型记录实际 taxel/三角形数量、耗时、CPU/内存及环境，不将
目标刷新率当作实测性能。双手验证只使用实际包含双手的可用 URDF。
无硬件时明确标注真实 live 未验证，但真实接收实现及合成路径仍须交付。

SDK 验证取决于启动工具的 Python 环境；不预设本机 SDK 构建产物、
旧解释器或模块路径。条件具备时执行真实 SDK 构造、编码和接收验证；
条件不具备时标注未验证，不另建环境、安装另一份 SDK 或构建 SDK 来
满足验收，不以合成数据或 mock 宣称官方验证通过。

交付复用启动 Python 环境的工具、Isaac Capture 环境依赖声明更新、基于
现有 model/ URDF 的 metadata 示例、可验证的导出产物、逐步使用说明
（含已有环境中的依赖准备和实时动态可视化）及面向开发者的实现与数据流
技术说明；不交付工具专用环境或锁文件。实现、界面文本和交付文档均使用英文。

```
