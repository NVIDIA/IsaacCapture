<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

<!--
  MANDATORY FOR ALL AUTOMATED CODING AGENTS (Cursor, Copilot, etc.):

  Before modifying OR creating ANY file under this repository, you MUST use your
  file-reading tools to load EVERY AGENTS.md on the directory paths you will touch
  (see "CRITICAL — mandatory preflight" below). This is NON-OPTIONAL. Skipping it,
  or relying only on chat context without reading those files, is incorrect.

  Before you treat work as finished, you MUST run pre-commit as in
  "Pre-commit — match CI before you stop" and fix all failures. Declaring the task
  done without that run is incorrect.

  If pre-commit/CI fails, or the user corrects you, or you repeat the same class of
  mistake, you MUST record the lesson in AGENTS.md or source comments per
  "Mandatory learning loop" in the same session—not only when the user asks.

  Repository owners: keep this block at the top of this file; do not delete it.
-->

# IsaacTeleop — agent notes

## CRITICAL — mandatory preflight (do not skip)

**Hard requirement:** Do **not** edit or add code under `IsaacTeleop/` until you have **read** (e.g. with the Read tool) **every** relevant `AGENTS.md` file as defined below. This is **not** optional, **not** "only for large tasks," and **not** satisfiable by inferring from prior conversation. If you have not read those files in **this** session for **this** task, stop and read them first.

**You must:**

1. List the **directories** you expect to edit or create files under (e.g. `src/core/live_trackers/cpp/`, `src/core/deviceio_base/cpp/inc/…`).
2. For **each** such directory, **read** **`AGENTS.md` in that directory** if it exists.
3. Walk **up** toward the **IsaacTeleop repo root** and **read** **`AGENTS.md` in every ancestor directory** that has one (e.g. `live_trackers/` → `src/core/` → **`IsaacTeleop/AGENTS.md` (this file)**).

**You must not** assume that reading the single "closest" `AGENTS.md` is enough. **Multiple** `AGENTS.md` files can apply to one change (e.g. both `live_trackers` and `deviceio_base`).

**Listing every `AGENTS.md` in this repo** (no curated table here—add new files under the tree without editing this document):

```bash
# Run from the IsaacTeleop repository root (the directory that contains this file):
find . -name AGENTS.md ! -path '*/.git/*' | sort
```

If you cannot run a shell, use your search/glob tools on the pattern `**/AGENTS.md` from the same root. That inventory is for orientation; you must still **read** every file that applies to the directories you will touch (steps 1–3 above).

Optional context index: [`src/core/AGENTS.md`](src/core/AGENTS.md) (also on the ancestor walk—read it when working under `src/core/`).

## CMake & include structure

When you create or edit a `CMakeLists.txt`, add or move a target, place a header,
write `#include` directives, or restructure directories, follow the repo's
canonical CMake/include rules in **[`cmake/cmake-structure.md`](cmake/cmake-structure.md)**.
Read that file before touching build files, header placement, or include paths.

Per-tool shims auto-load the same doc for Claude Code, Cursor, Copilot, and
CodeRabbit (`.claude/skills/cmake-structure/`, `.cursor/rules/cmake-structure.mdc`,
`.github/instructions/cmake-structure.instructions.md`, `.coderabbit.yaml`); they
only point here — edit the rules in the doc, not the shims.

## Shell scripts

- In **bash** scripts (`#!/bin/bash`, `#!/usr/bin/env bash`, or files `source`d
  only by bash), use **`[[ ... ]]`** for conditional tests, never the POSIX
  single-bracket **`[ ... ]`**. `[[` is safer (no word-splitting or glob
  expansion on unquoted operands) and more feature-rich; SonarQube/ShellCheck
  (SC2292) flag the single-bracket form.
- **Caveat:** `[[` is a bash builtin and is **not** POSIX. Before converting,
  confirm the script's interpreter is bash — a genuine `#!/bin/sh` (dash) script
  must keep `[ ... ]`. Check the shebang (and, for sourced files, who sources
  them) first.

## Logging — one tree, one place to configure it

Every process in this repo, Python and C++, feeds a single logger tree rooted
at `isaaccapture`. These rules apply wherever you emit output, not only inside
the logging packages.

- **Name every logger under `isaaccapture.`** In Python that means
  `logging.getLogger(__name__)` inside `src/python/isaaccapture/` (where
  `__name__` already starts with `isaaccapture.`) and an explicit
  `logging.getLogger("isaaccapture.<area>.<module>")` anywhere else, including
  `examples/`. In C++ it means
  `isaaccapture::Logger::get("isaaccapture.<module>.<ClassName>")`. Dotted names
  **are** the hierarchy: a bare name like `"robot_viz"` is a sibling of
  `isaaccapture`, not a descendant, so the console and file handlers attached to
  the root of this tree can never see its records.
- **Never call `logging.basicConfig()`**, and never attach your own handler to
  the `isaaccapture` logger or to the Python root logger. `basicConfig()`
  installs a handler on the *root* logger, and nothing here sets
  `propagate = False`, so from that call onward every `isaaccapture` record is
  emitted twice — once in the shared line format and once in yours.
  **The one exception is a program that imports no `isaaccapture` at all**, and
  is meant to keep it that way: there are no handlers of ours for it to
  duplicate, and reaching for `logging_config` would add the dependency the
  program exists without. `examples/camera_viz/camera_streamer.py` is the
  case — a sender-only camera box runs it with no CUDA/Vulkan/OpenXR runtime
  installed, so `import isaaccapture` would fail outright. Keep the logger
  names under `isaaccapture.` anyway; the naming rule above is a convention,
  not a runtime coupling.
- **Configure through the public API instead:** `logging_config.set_console_level()`,
  `set_console_filter()`, `set_logger_colors()`. A `--verbose` flag should call
  `set_console_level("debug")`, not build a handler.
- **`print()` / `std::cout` are for deliberate terminal UX only** — CLI usage
  text, interactive prompts, operator banners, progress lines a log file would
  ruin. Diagnostics go to a logger. Existing raw-output sites that survive in
  `src/plugins/*/main.cpp`, `cloudxr/oob_teleop_*.py` and similar are that
  deliberate kind; do not "migrate" them, and do not add new ones for
  diagnostics.
- **One operator message, one channel.** The line rejecting an argument and
  the usage block printed straight after it are one message; splitting them
  between a logger and `std::cout` shows the operator half of it, and under
  `ISAACCAPTURE_LOG_SOCKET` — where `local_sinks()` gives a forwarding sink and
  no console sink — none of the logged half. Judge such a site by the message
  it belongs to, not by the call on its own line.
- **A log message some machine greps for is an interface.** Before rewording
  or re-homing one, `grep -rF` the phrase over `.github/`, `docs/`, `scripts/`
  and `tests/`. `.github/workflows/build-ubuntu.yml` alone waits on three
  literals and fails fast on eight more, and a miss there does not look like a
  broken marker — the job waits out its timeout and blames the wrong thing.
  Note what moving a class name from the message into the logger name costs:
  the two halves are then separated by the level and pid columns, so no fixed
  string spans them any more.
- **Six environment variables are the whole external contract.** The first
  four are read identically by both halves — change one and change both, or
  the two stop agreeing: `ISAACCAPTURE_LOGGING` (see the next rule),
  `ISAACCAPTURE_LOG_DIR` (where log files land),
  `ISAACCAPTURE_LOG_LEVEL` (the console threshold every process starts at, the
  Python leader's included; the six level names and the numeric form, nothing
  else — spdlog's `warn`/`err` spellings are deliberately not accepted, because
  Python has no entry for them) and `ISAACCAPTURE_LOG_SOCKET` (set by the session
  leader; its presence is what makes a process forward instead of owning
  handlers). The last two belong to the Python half alone:
  `ISAACCAPTURE_NATIVE_CAPTURE=off` switches off `capture_native_output()`'s
  rebinding of *the host's* fd 1 and fd 2 (a process isaaccapture launches still
  gets the capture file — its descriptors are not the host's), and
  `ISAACCAPTURE_NATIVE_CAPTURE_FILE` is written by the leader and read by
  processes with no interpreter so they can point their own stdio at the same
  file. Do not invent a seventh.
- **`ISAACCAPTURE_LOGGING=off` switches this whole system off**; unset or any
  other value leaves it on. Off means none of it runs — no handler, log file,
  socket, native capture or thread of its own — and C++ loggers write every
  level to stderr. A call site whose output would otherwise vanish keeps its
  pre-system code (`print()`, `basicConfig()`, its own handler) behind the
  check; only that off-path code is exempt from the rules above, since nothing
  of ours is attached for it to duplicate.
- **Never rebind the host process's fd 1 or fd 2 outside a scope.** This is a
  library its host imports; its descriptors are not ours. Output that no logger
  can reach — vendor code that formats its own lines onto a descriptor — is
  captured either inside `logging_config.capture_native_output()`, which
  restores what it found, or in a process this library launched, whose
  descriptors *are* ours to set. There is deliberately no process-wide mode and
  no import-time `dup2()`; do not add either back.

Subsystem-internal rules live with the code:
[`src/python/isaaccapture/logging_config/AGENTS.md`](src/python/isaaccapture/logging_config/AGENTS.md)
and [`src/core/log_bridge/AGENTS.md`](src/core/log_bridge/AGENTS.md). Read the
relevant one before changing either half.

## Comments and docstrings — say it once, briefly

Comments earn their place by recording what the code cannot: a constraint, a
measurement, a trap someone already paid for. They are not a place to narrate.

- **Default to a few lines.** A block comment over ~8 lines, or a docstring
  over ~6, needs a reason. If the explanation is genuinely long, it belongs in
  a `README.md` / design doc that a reader can skip, not inline where they
  cannot.
- **State the fact, drop the argument.** "MuJoCo rewrites meshes into their
  inertial frame, so STL axes need `mesh_pos`/`mesh_quat`" beats three
  paragraphs re-deriving why. Keep measured numbers and file/line references;
  cut the prose around them.
- **Do not narrate history.** "An earlier revision did X and it was wrong" is
  what `git log` is for. If the wrong approach is tempting enough to warn
  about, write the warning as a rule — "do not derive this from the mesh" —
  not as a changelog.
- **No emphasis-by-shouting.** Occasional caps for a single load-bearing word
  is fine; whole sentences in caps are noise once every third comment has them.
- **One statement of a fact, in one place.** If a constraint is already in the
  code, the test name, or an adjacent doc, do not restate it.

This applies to `#`/`//` comments, docstrings, and comment blocks in
`CMakeLists.txt`, `pyproject.toml`, YAML data files and scene XML.

## Commits — DCO sign-off for AI-drafted commits

Any commit whose message is drafted or edited by an AI agent **must** include
a `Signed-off-by:` line (Developer Certificate of Origin). Pass `-s` /
`--signoff` to `git commit`, or add the line manually:

```
Signed-off-by: Your Name <your@email.com>
```

The name and e-mail must match the committer's git identity
(`git config user.name` / `git config user.email`).

A `commit-msg` pre-commit hook enforces this. Install it **once per clone**
(in addition to the usual `pre-commit install`):

```bash
pre-commit install --hook-type commit-msg
```

## Remote pushes

- Before pushing, compare every URL from
  `git remote get-url --push --all <remote>` with the user-specified owner and
  repository. Do not infer the destination from the account, remote alias, or
  branch name.
- Before updating an existing pull request, verify its exact head owner,
  repository, and ref. A similarly named branch does not update that pull
  request.

## Pre-commit — match CI before you stop

- From the **IsaacTeleop repo root** (this directory), run pre-commit and **fix all failures** before you treat a change as finished (do not only rely on “should pass” reasoning).
- **Use the same hook set as GitHub Actions:** `.github/workflows/pre-commit.yaml` runs pre-commit with **`SKIP=check-copyright-year`**. Mirror that locally:

  ```bash
  SKIP=check-copyright-year pre-commit run --all-files
  ```

- **`--all-files` means all *tracked* files.** pre-commit enumerates through
  `git ls-files`, so a file you have created but not yet `git add`ed is skipped
  and the run passes for the wrong reason — REUSE and `ruff format` alike.
  Stage first (`git add -A`), then run the hooks.
- **REUSE:** files covered by the REUSE hook need **`SPDX-FileCopyrightText`** and **`SPDX-License-Identifier`** in the form the repo already uses (for example the HTML comment block at the top of `README.md` also applies to **`AGENTS.md`** and similar docs).
- **C++ formatting is enforced by CI, not pre-commit.** The hook set runs `ruff` for Python but does **not** run `clang-format`; CI (`build-ubuntu.yml`) installs **`clang-format-14`** and rejects unformatted C++ as `-Wclang-format-violations`. Before pushing, format touched C++ with the system `clang-format` (match CI's version 14) and verify:

  ```bash
  clang-format -i $(git diff --name-only main -- '*.cpp' '*.hpp' '*.h' '*.cc')
  clang-format --dry-run --Werror $(git diff --name-only main -- '*.cpp' '*.hpp' '*.h' '*.cc')
  ```

- If a hook failure shows **missing or non-obvious repo policy** (not a one-off typo), you **must** add a **short** reminder under **Mandatory learning loop** rules to the right `AGENTS.md` or adjacent **`//` comments** so the next run does not repeat it—unless it is already documented.

## The wheel SBOM

Published wheels carry a contents SBOM and license evidence derived from the
build itself. There is **no checked-in dependency registry** and adding one would
be a second source of truth — declare a dependency the way you already would and
it appears in the next wheel's inventory.

Two things fail the Release build rather than shipping a gap, and neither is
fixed by excluding a file or writing down an assumption:

- **a wheel member the build cannot explain** — teach the collector that route;
- **a redistributed component with no license text** — obtain the terms.

The collector is a project of its own: `uv run --project scripts/sbom
isaaccapture-sbom ...`. Its dependency pins live in `scripts/sbom/pyproject.toml`
and nowhere else — never restate them as `--with` flags in a workflow or a doc.

**Read [`docs/source/references/sbom.rst`](docs/source/references/sbom.rst)
first** if you are changing anything under `scripts/sbom/` or
`tests/python/sbom/`, changing how a dependency reaches a wheel, or touching the
wheel-packaging steps in `build-ubuntu.yml`. It covers discovery, license
matching and the maintenance rules; nothing here repeats them.

## Mandatory learning loop (AGENTS.md and comments)

**Hard requirement:** When **any** of the following happens, you **must** complete steps 1–3 **before** you end the session or move on as if the work were complete:

1. The **user** is dissatisfied or corrects your approach (wrong layer, scope, style, or a fix that does not stick).
2. **Pre-commit** or **CI** fails on something you or another agent could hit again (linters, REUSE/SPDX, formatting, policy hooks, etc.).
3. You **repeat** the same **category** of error after a correction.

**You must:**

1. **Distill** what went wrong into a **short, reusable pattern** (a rule or boundary, not a chat transcript).
2. **Update** the right artifact in the **same working pass** as the fix: prefer the **most local** `AGENTS.md` for package- or subtree-level rules; use the **repo root** `AGENTS.md` only for expectations that span the whole tree; put **volatile or line-specific** detail in **`//` comments** next to the code.
3. Respect **scope vs `main`** (below): only add `AGENTS.md` bullets for **this branch’s delta** to **`main`** (or the agreed base), not for behavior that already exists on the base branch.

**You must not** skip documentation updates because the user did not say “update AGENTS.md”—failed checks and repeated mistakes **trigger** this loop automatically.

**What belongs in `AGENTS.md`**

- **High-level** expectations: boundaries between layers, what to avoid, naming or structural conventions, CMake/include policy at a glance, “always / never” rules that stay true across refactors.
- **Style of work**: how minimal to keep diffs, when to ask for clarification, how this subsystem should relate to OpenXR/schema/deviceio, etc.
- For build-only requests, use the documented CMake flags and environment first (for example `-DISAAC_TELEOP_PYTHON_VERSION=...`) rather than patching build files to select options.

**What does *not* belong in `AGENTS.md`**

- **Low-level or volatile detail**: exact call sequences, field-by-field semantics, long checklists tied to one function, anything that will go stale the next time the code moves.
- Put that in **comments in the source files** where the behavior lives (short `//` notes: intent, invariants, or “do not X because …”).

**Scope — document what *this* branch changes, not main**

- New or updated bullets should capture **learnings from the delta** between **this branch (or commit)** and **`main`** (or whatever long-lived base you are targeting—release branch, parent MR, etc.).
- Do **not** restate facts that already hold on **`main`** just because this branch touches nearby files. If behavior landed in the **base** history, it belongs in **code comments** next to that code or in **main’s** docs—not as “new” guidance in an `AGENTS.md` introduced only to support a smaller follow-up change.
- Example: schema- or API-level choices that merged **before** this branch are **out of scope** for `AGENTS.md` edits tied to this branch; they add churn and read like stale noise after merge.
