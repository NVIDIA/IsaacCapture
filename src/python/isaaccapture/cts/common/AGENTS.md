<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `cts/common/`

**CRITICAL:** complete the mandatory `AGENTS.md` preflight in [`../../../../../AGENTS.md`](../../../../../AGENTS.md)
before editing here. A change here reaches every schema's checker, so also read the
`AGENTS.md` of each checker under `../*/checker/`.

`cts_common` is what the per-schema checkers share: file-order MCAP reading by
schema name, the accumulator model, the envelope and payload checks, verdict
aggregation, `Mark`, the label sidecar, and the viewer's `Sample`, `Track`, `Skeleton`
and status-list HTML.

- **Schema-agnostic.** No record type, joint table, profile or check group lives here;
  `tests/test_boundaries.py` rejects the names. A shared check carries no `group` — each
  schema package subclasses it to set one.
- **Same dependency floor as the checkers:** `mcap` and `flatbuffers`, no
  `isaaccapture`. viser only in `panel/skeleton.py`.
- **The oracles import nothing from here.** Their quaternion arithmetic is duplicated on
  purpose; the full-body and hand `test_boundaries.py` assert it.
- **Changing a shared check changes two verdicts.** Run both checkers' suites.
- **After moving code in or out of here, run pre-commit, not only the tests.** Under
  `from __future__ import annotations` a name used only in a type hint is never
  evaluated, so a dropped import passes every test and fails ruff (F821).
