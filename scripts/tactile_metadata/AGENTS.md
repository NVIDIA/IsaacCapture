<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Tactile metadata editor agent notes

- 工具设计必须明确界面操作、持久化字段、坐标转换、绑定生成和 SDK 接入，不能把需求清单作为完整设计交给实现代理。
- 设计按用户界面与功能、依赖、代码架构、模块内部实现及约定、代理执行规范分类；正确性、性能和扩展性归入相应类别，不与设计层级并列。代理执行规范与工具设计分开组织。
- 用户要求完全重写时，从用户操作与数据流重新建立规格和实现提示词，不以搬运原段落或更换标题代替重写。
- 多语言文档以 REUSE 实际识别的授权信息为准；钩子生成 `.license` 附属文件时，核对并随文档一起暂存，再执行完整检查。
- 实现以本目录的 `DESIGN.md` 为依据；协议事实以仓库 schema、绑定和共享校验器为准。发现冲突时提供具体代码证据，并同步修正设计与实现提示词。
- 首版的手套规则留在手部适配层；左右配置独立，通用编辑状态不使用 `HandJoint` 枚举作为内部标识。
- 评审改变输入范围、术语或编辑实体时，同时更新界面、持久化、算法、验收与实现提示词；已取消的产品概念不能留作隐藏的数据模型。
- 对设计文档的结构与表达要求应直接落实到目标文档；页面布局按区块列出具体控件，不以聊天中的补充说明代替文档修改。
- Keep URDF driving separate from taxel deformation: HandPose supplies skeleton and LBS poses, while live URDF motion uses JointState only. Remove obsolete driver controls, persisted options, algorithms, and acceptance cases together.
- Draft forms always read the editable Document. Validation reports errors without blocking draft edits or replacements; received metadata replaces only explicitly selected sections and never becomes a form source by changing the visualization.
- Preserve raw skinning slots and weights through editing and export; do not coalesce repeated segment indices. Distinguish receiver failures from sender behavior, and keep all editor operations local and undoable.
- SDK process isolation specifies where SDK classes are called; document the actual tracker/session calls required for live reception as well as export encoding.
- Keep tool implementation inside this subtree; add missing dependencies to Isaac Capture's existing requirements and matching package metadata. Run checks only from this directory or its descendants; never run repository-wide fixing hooks that can change other directories.
- Explain this standalone tool's setup from the user's prerequisites: define external libraries before packaging terms, and distinguish requirement files, generated version records, and installation commands. Identify inspected environments by their Python executable path; distinguish declared requirements from packages actually installed there.
- Reuse the launching Python environment and its SDK through sys.executable. Do not create tool-specific project metadata, lockfiles, environments or SDK installations, and do not infer runtime requirements from old local build artifacts.
- Missing packages in the agent's inspection environment limit verification; report the unavailable checks. Do not turn verification requirements into proposals to install dependencies, provision environments, or substitute interpreters; maintain Isaac Capture's shared dependency declarations instead.
- Treat missing library convenience callbacks as implementation work. A minimal local browser-event bridge that preserves the specified interaction needs no separate approval; ask only when a demonstrated limitation requires changing product behavior or exceeding authorized boundaries.
