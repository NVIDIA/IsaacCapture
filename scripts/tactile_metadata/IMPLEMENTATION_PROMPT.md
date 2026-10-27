<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Tactile Metadata Editor 实现提示词

将以下提示词交给能够读取本仓库的实现代理。完整规格见同目录 [DESIGN.md](DESIGN.md)。

```text

请实现 /Users/lotusl/Codes/IsaacTeleop/scripts/tactile_metadata/DESIGN.md
定义的 Tactile Metadata Editor。All implementation should use English instead of Chinese.

When implementation, make sure to only modify folder /Users/lotusl/Codes/IsaacTeleop/scripts/tactile_metadata and its decendent files. It's where your implementation should fully locate. Don't write to any other paths, except for /tmp and your own workspace, and the IsaacTeleop folder's library config only to add the library dependencies as guided. Don't connect to any remote computer or write to any remote end. You're allowed to read codes in the whole ~/Codes/IsaacTeleop. Don't run pre-commit or other whole-repo formatter/tests. It's OK to run local formatter/test within tactile_metadata.

先完整读取设计、仓库根及所有修改路径适用的 AGENTS.md，检查已有改动。
这是实现任务，按已确定的规格工作。编码前完整了解领域、数据流及已有
能力，以现有 schema、公开绑定、共享校验函数和 tracker/session 核实
协议。

The following product decisions are settled and must be reflected throughout the implementation:
1. Weight editing: after a slider or numeric-field edit, replace that slot and normalize all four weights by their sum, including the edited slot. Reject negative weights. Reject an edit whose resulting row contains non-finite or negative values or has a zero sum, preserving the previous row. Apply normalization independently to each selected taxel. Preserve slot order and repeated indices. Loading, explicitly recording received metadata, checking, saving, and exporting preserve raw weights and report invalid values without silently normalizing them.
2. Metadata persistence: saving, backups, and export write only tactile metadata values using the fields, nested structure, and union type discriminators defined by src/core/schema/fbs/tactile_metadata.fbs. The only top-level JSON fields are geometry_mapping_type, geometry_mapping, segment_source_type, and segment_source. Use one schema JSON representation for saving, reopening, and JSON export; optional SDK binaries encode the same values. Do not add an editor extension, tool-specific fields, project/configuration files, .editor.json, internal-ID correspondence tables, or saved check reports. Read with json.load(), perform the array and enum conversion required by the existing SDK constructors, and encode through LBS, HandSkeletonRef, TactileMetadata, and to_bytes() in the isolated SDK subprocess. Do not invent another metadata format or treat this conversion as an unresolved product decision.
3. Scope: metadata/force alignment is the producer's and user's responsibility. Display the current arrays in protocol row order and retain structural, count, and validity checks. Do not maintain source-row mappings, correspondence across reordering, alignment baselines, or force permutations to compensate for metadata edits. Switching visualization sources does not establish or repair force alignment. Persist taxel output order through array row order, without saving internal IDs or temporary numbering.
4. URDF display: all links use fixed opacity 0.6. Do not add per-link visibility or opacity controls or state. Highlight associated links through color without changing opacity; retain mode-level model visibility.
5. Session state: URDF/package paths, q0, wrist calibration, segment/link associations, hand assignments, live-source and joint mappings, internal IDs, and display settings exist only in the current session. Reopening restores only metadata values and array order; the user assigns the hand through the GUI or --sides and sets up the URDF associations and calibration again when needed. Unsaved state tracks metadata values only. Incomplete saves use the same schema fields; show draft status, validation results, and revisions in the GUI or CLI output, never in metadata or companion files. Validate formal exports separately from serialization. Do not implement separate editable/numerical formats, session restoration, or a --report file output.

Before implementation, do three analysis:
1. whether there's any unresolved 规格矛盾、错误假设、风险、缺失功能或 SDK 能力缺口.
2. whether the implementation can be scoped to 2000 lines of codes or fewer (this impentation shouldn't delete or modify any existing lines, only will add lines).
3. whether any operation would require my manual permission.
If finding any of the gaps above, stop and clarify the gaps.

Over-engineering is absolutely forbidden. 2000 lines is the hard limit. If ever you exceed this number, it's considered to be a failure, and you need to remove all your changes and analyze what you did wrong.
Now redo the task with the new 2000 line limit and strictly obey the minimal implementation rule.

```
