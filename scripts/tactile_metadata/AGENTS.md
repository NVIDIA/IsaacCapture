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
- 评审替代性 UI 规格时，以当前稿为产品范围，只把 schema/API 约束视为阻塞；不要把旧功能、状态增强或可接受的瞬态覆盖升级为要求，并逐字核对交互手势。
- 对设计文档的结构与表达要求应直接落实到目标文档；页面布局按区块列出具体控件，不以聊天中的补充说明代替文档修改。
- Keep URDF driving separate from taxel deformation: HandPose supplies skeleton and LBS poses, while live URDF motion uses JointState only. Remove obsolete driver controls, persisted options, algorithms, and acceptance cases together.
- Live Preview segments come only from `is_valid=True` HandPose joints in HandJoint order; invalid joints are absent, and an invalid WRIST yields an empty segment list because wrist-relative conversion is unavailable. The selected metadata source controls only taxel bindings and never seeds or overwrites Preview segments.
- Preview deformation is per taxel: omit invalid taxels and report their row errors without suppressing valid taxels. Preserve original row indices when matching live force values. Apply Taxels and Apply ALL copy exactly the valid subset returned for 3D rendering and never require every taxel to be valid.
- Preview apply operations remain independent and available despite taxel errors. Error reasons persist in the fixed Info area after disconnects and unrelated Preview actions.
- Draft forms always read the editable Document. Validation reports errors without blocking draft edits or replacements; received metadata replaces only explicitly selected sections and never becomes a form source by changing the visualization.
- New documents initialize every OpenXR HandJoint segment in the neutral hand pose, mirroring the left hand before visualization-space conversion; loading replaces these defaults instead of merging with them.
- Preserve slot order and repeated segment indices. Normalize all four weights only on explicit user weight edits and reject negative edit values; loading, recording received values, checking, saving, and exporting preserve raw weights. Distinguish receiver failures from sender behavior, and keep all editor operations local.
- SDK process isolation specifies where SDK classes are called; document the actual tracker/session calls required for live reception as well as export encoding.
- Keep tool implementation inside this subtree; add missing dependencies to Isaac Capture's existing requirements and matching package metadata. Run checks only from this directory or its descendants; never run repository-wide fixing hooks that can change other directories.
- Explain this standalone tool's setup from the user's prerequisites: define external libraries before packaging terms, and distinguish requirement files, generated version records, and installation commands. Identify inspected environments by their Python executable path; distinguish declared requirements from packages actually installed there.
- Reuse the launching Python environment and its SDK through sys.executable. Do not create tool-specific project metadata, lockfiles, environments or SDK installations, and do not infer runtime requirements from old local build artifacts.
- Missing packages in the agent's inspection environment limit verification; report the unavailable checks. Do not turn verification requirements into proposals to install dependencies, provision environments, or substitute interpreters; maintain Isaac Capture's shared dependency declarations instead.
- Use Viser's standard controls and public callbacks, with document changes serialized through State. Apply the same explicit weight-normalization rule to sliders and numeric fields.
- The selected-object transform control enables only XYZ axis translation. Serialize drag phases through the event queue; single selections use object axes, multi-selections use visualization-space axes, and the same translation delta applies to every selected object.
- Selection side-highlighting resolves bindings by HandJoint name and includes only slots with positive weights. Keep `SIDE_HIGHLIGHT_COLOR` distinct from `SELECTED_COLOR`; selection color wins on selected objects and validation error color wins on invalid taxels.
- When product placement says top/before/below and any sibling uses explicit Viser `order`, assign explicit order to every constrained sibling; source creation order does not override mixed default and explicit ordering.
- Route action and validation errors to the fixed, non-movable client-side Info area at the bottom of the 3D viewport, not a Viser panel, the right menu, per-button, or top-menu markdown. The Selected Object panel is a floating window on the left, not a dock. UI sections marked `(spannable)` in the product design and all nested entity/joint/binding folders start collapsed.
- Client-side helpers may hide only their exact bridge input and label; never hide an ancestor container that can contain the entire Viser control panel.
- Preserve focused numeric drafts such as `1.` client-side because Viser ignores partial numeric values and controlled rerenders otherwise erase the decimal point before the user finishes typing. Restore through the native input value setter and dispatch an `input` event; a raw DOM assignment leaves React/Mantine state stale and breaks the next digit.
- The right menu is a flex column: General is fixed, the Edit tab group fills the remaining height, and only the active tab body scrolls. Scope DOM styling to the root containing the General folder and URDF/Segments/Taxels dock group.
- Drain cross-thread UI events with nonblocking reads and `queue.Empty`; `Queue.empty()` is not a synchronization boundary. Give no-op-looking actions such as creating an empty document visible completion status.
- Call `/edit`, `/preview`, and `/saved` Viser scene-node groups, not image layers; this tool has no image-layer abstraction.
- During live redraw, update the active scene-node group in place. Hide only groups that should be inactive; never hide and immediately re-show the active group each frame.
- Keep metadata/force alignment with the producer and user; do not add source-row mappings, alignment baselines, force permutations, or list-order controls to the editor.
- JSON follows tactile_metadata.fbs and the existing bindings. Treat array and enum conversion for encoding as routine implementation rather than a separate product-format decision.
- Persist only schema-defined tactile metadata fields. Keep model configuration, calibration, source mappings and internal IDs in session state; audit actual save/export output rather than inferring persistence requirements from editing needs.
- Simplify persistence by deleting unsupported branches in place; do not relocate serialization into Document or introduce new state abstractions for a narrower file format.
- For local editor checks using a cached Ruff executable, match the version in the repository hook configuration; older binaries may not understand the shared lint rules. Keep every check and formatter target inside this subtree.
- Keep the GUI a thin adapter over shared document, geometry, storage and SDK operations. Prefer explicit native Viser widgets and one reusable property inspector; avoid widget-type inference, duplicate field editors in dialogs, and rebuilding scene nodes on every frame. Moving logic between files alone does not reduce implementation size.
- Segment cubes use flat shading with cast and receive shadows disabled; do not restore smooth vertex-normal gradients to semantic markers.
- Use URDF base opacity 0.4 multiplied by the rendered dataset group's opacity, with color-based link highlighting; do not add per-link visibility or opacity controls or state.
- Under a hard code budget, finish and count a complete executable slice before expanding the UI. Refactoring dispatch or moving functions is not evidence that the implementation fits; verify the formatted total and retain budget for integration fixes.
- Treat metadata wrist coordinates, Viser scene coordinates and camera screen orientation as distinct; diagnose an upside-down display from the scene/camera mapping, not URDF alignment.
- Separate OpenXR schema/reference-space contracts from provider basis conversion; verify the plugin conversion before claiming injected HandPose follows OpenXR joint axes.
- Rebase both the raw live skeleton and metadata deformation by the same full inverse WRIST pose; never reattach the world-space wrist transform when the live view is wrist-relative.
- Render viewport direction indicators in client-side screen space from the render camera; perspective scene nodes and server camera callbacks introduce distortion, delay and jumps.
- When reporting a captured protocol frame, include every row in protocol order; a summary may follow but must not replace the requested raw frame.
