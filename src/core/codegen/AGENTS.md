<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `codegen`

**CRITICAL (non-optional):** Before editing this package, complete the mandatory **`AGENTS.md` preflight** in [`../../../AGENTS.md`](../../../AGENTS.md) (read every applicable `AGENTS.md` on your paths, not just this file).

This package generates the schema-based tracker stack from
[`../deviceio_trackers/trackers.toml`](../deviceio_trackers/trackers.toml) +
[`../deviceio_trackers/defaults.toml`](../deviceio_trackers/defaults.toml). `manifest.py` resolves
`%placeholder%` values, `templates.py` derives per-entry type/file names, `template_renderer.py` loads
`templates/` bodies with `@KEY@` substitution, and `generate_trackers.py` renders the sources
and the `.inc` fragments.

## Boundaries

- **Stdlib only, and this one is load-bearing.** The generator runs at CMake configure time from
  whatever interpreter `Python3_EXECUTABLE` points at — a bare uv-managed CPython with no project
  venv and no site-packages. A third-party import here is not a dependency, it is a configure-time
  crash. That is why the manifests are TOML (`tomllib` is stdlib on 3.11+, the supported floor) and
  why there is no Jinja2.
- **Generation happens at configure time, not build time**, because CMake needs the source list up
  front. Anything that changes the output (both TOMLs, every `.py` here, and every template under
  `templates/`) must stay listed in
  `CMAKE_CONFIGURE_DEPENDS` in [`../../../cmake/GenerateTrackers.cmake`](../../../cmake/GenerateTrackers.cmake),
  or incremental builds will silently use stale output.
- **Emitted files use compare-before-write.** Content is rewritten only when it differs from the
  on-disk copy, so no-op configure runs keep mtimes stable and avoid needless rebuilds. Output
  lands under `${CMAKE_BINARY_DIR}/generated/trackers/` (explicit cmake source lists — not a
  filesystem glob of that tree).
- **Optional `--prune-stale`** deletes previously generated files this run did not emit (suffix +
  `AUTO-GENERATED` banner; requires `.isaac_teleop_tracker_codegen`). CMake configure passes this
  flag so renamed or removed trackers cannot leave an old header satisfying a stale `#include` on
  the generated include path. Omit the flag for manual/dry runs that should only rewrite.
- **Prefer a manifest key over a template branch.** If one tracker differs, give it an override key;
  add a new manifest `shape` only when the control flow genuinely differs. A shape that makes the
  templates hard to follow is a signal to leave that tracker hand-written and say so in
  [`../deviceio_trackers/AGENTS.md`](../deviceio_trackers/AGENTS.md). Optional `header` overrides the
  derived file stem (and thus `*_base` / `live_*_impl` / `replay_*_impl`) when `name` alone would
  invent a different public `#include` path.
- **`direction` must be `pull` or `push`** — validated before defaults merge so a typo cannot silently
  pick up the wrong overlay. **`shape=single_collection` requires `record=true`** for `direction=pull`:
  those templates always emit MCAP channel/viewer constructors, while the factory only omits them
  when `record=false`. Non-recording multi-sample readers (e.g. `HapticCommandReaderTracker`) stay
  hand-written — there is no generated `multi_endpoint` shape.
- **Emit code that already satisfies the layering rules** of the package it lands in — generated
  `deviceio_trackers` facades must not include OpenXR headers, generated live impls own the
  `SchemaTracker`, and so on. The generator is not exempt from a package's `AGENTS.md`.

## Working on the templates

- **Per-tracker C++ bodies** live under [`templates/`](templates/) as `*.hpp.template` /
  `*.cpp.template` (`pull/` and `push/`) plus `*.template` fragments under `fragments/` for
  factory/pybind/traits. Direction-specific fragments use `_pull` / `_push` suffixes (e.g.
  `pybind_pull.template`, `live_factory_push.template`, `recording_traits_pull.template`); shared
  pieces are `live_try_create.template` and `replay_try_create.template`. Edit those for structure and includes; use `@KEY@` placeholders filled by
  `template_renderer.template_values()`. [`templates/.clang-format`](templates/.clang-format)
  sets `DisableFormat: true` so editors that map `*.cpp.template` to C++ for highlighting cannot
  mangle `@KEY@` on save. Conditionals stay in Python (pick which template to load)
  or as `#if @FLAG@` for optional fragments within one shape (e.g. `@HAS_TENSOR_IDENTIFIER@`). Do
  **not** merge `push/` with `pull/` via shared `#if`/`#else` — those control flows are different
  enough that separate templates stay clearer. `*.hpp.template` / `*.cpp.template` carry SPDX
  headers for REUSE; `template_renderer` strips them before prepending the emit-time COPYRIGHT /
  AUTO-GENERATED banner.
- **One-line aggregators** (includes, dispatch rows, factory decls, cmake source lists, Python
  `__all__`) remain `yield` loops in `generate_trackers.py` joined with `"\n"`.
- Generated C++ is **not** covered by the format gate (`cmake/ClangFormat.cmake` globs only
  `src/` and `examples/` and excludes `build/`), so it does not need to be clang-format clean — but
  it must compile. Read the emitted file under `${CMAKE_BINARY_DIR}/generated/trackers/` when a
  template change fails to build; the compiler error points at generated text, not at the f-string.
- Generated `#include`s must be written as the **consumer** sees them (`<deviceio_base/...>`,
  `<live_trackers/schema_tracker.hpp>`), not as the in-tree relative paths the hand-written files
  use, because generated sources sit in a different directory.
- Run `python test_manifest.py` after touching resolution rules, and
  `generate_trackers.py --print-resolved` to see the fully expanded manifest before blaming a
  template.
- A **full build is the only real test** of a template change: it is what proves the emitted
  facade, impls, and every `.inc` fragment still agree with the hand-written files including them.
