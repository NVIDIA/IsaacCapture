<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `schema`

**CRITICAL (non-optional):** Before editing this package, complete the mandatory
**`AGENTS.md` preflight** in [`../../../AGENTS.md`](../../../AGENTS.md) (read every
applicable `AGENTS.md` on your paths, not just this file).

## Read [`README.md`](README.md) before changing anything under `fbs/`

Every MCAP recording embeds the schema it was written under, and both the build and the
replay path compare that against what these schemas compile to now. An edit here can make
recordings already on disk unreadable, so treat `fbs/` as a published contract rather than
ordinary source.

[`README.md`](README.md) has the evolution rules and what to do when the conform test
fails. The short version:

- **Append, with a fresh `id`, and leave it optional.** Never renumber or reuse an id;
  `(required)` breaks every message recorded before the field existed.
- **Deprecate, never delete.** `(deprecated)` keeps the ids after it in their slots.
- **A `root_type` rename is a break**, even though no byte moves — the recording carries
  the old name and the reader matches it exactly.
- **Never change a `struct`** — not a field added, removed, reordered, or retyped. An
  *appended* struct field is the one break `flatc --conform` does not report, because every
  existing offset survives; `check_schema_compat()` catches it on the struct's size.
- **Never refresh `golden/` to clear a failing test.** The failure is the signal. Refresh
  only once breaking existing recordings is a decision you have made and stated.
