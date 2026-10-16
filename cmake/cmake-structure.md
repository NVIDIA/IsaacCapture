<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Isaac Capture CMake & include structure

Single source of truth for how CMake targets, directories, and `#include` paths
are organized in this repo. These rules apply to **all** coding agents and AI
reviewers (Claude Code, Codex, Copilot, Cursor, CodeRabbit, Greptile, …) as well
as humans. Apply them both when **authoring** code/build files and when
**reviewing** changes that touch CMake, headers, include paths, or layout.

> This file is the canonical copy. The per-tool config files
> (`.claude/skills/cmake-structure/`, `.cursor/rules/cmake-structure.mdc`,
> `.github/instructions/cmake-structure.instructions.md`, `.coderabbit.yaml`,
> and the root `AGENTS.md`) point here — edit the rules **here**, not in those
> shims.

## The six rules (non-negotiable)

1. **One CMake target per leaf directory.** A directory that defines a target
   defines exactly one (one library, one executable, or one test executable).
2. **Never use `../` in include paths** — not in CMake files, not in `#include`
   directives, not anywhere. If you reach for `../`, the layout or the target's
   include dirs are wrong.
3. **Use `"..."` with relative paths for a target's OWN includes:**
   ```cpp
   // foo.cpp - a target-root source
   #include "implementation.h"   // own private headers, next to cpp
   #include "inc/foo/core.h"     // own exported (public) headers, relative to cpp

   // foo.h - a public header at inc/foo
   #include "core.h"             // own exported (public) sibling headers
   ```
   Enforce this by **never** adding the target's own `inc/` or `.` to its own
   include paths. Own headers are always reached by relative `"..."` paths.
4. **Use `<...>` only for std and other modules' headers** found via *external*
   (interface) include paths.
5. **Prefix the module name to public includes:**
   ```cpp
   #include <foo/core.h>   // another module's public header
   ```
   This is enforced by exposing only the parent of the module dir:
   `target_include_directories(foo INTERFACE inc)` — so consumers must write
   `<foo/...>`. **Use `INTERFACE`** (not `PUBLIC`/`PRIVATE` for the exported
   path) to keep a target's own includes separate from what it exports.
6. **`inc/` holds only exported headers.** A header that no other target is
   meant to consume must **not** live in `inc/` — keep it next to its `.cpp`
   as a private header, reached by a relative `"..."` path. It follows that a
   **non-library leaf module** (an executable or a loadable plugin — it links
   *consumers* and exports nothing) has **no `inc/` directory at all**: every
   one of its headers is private and sits beside its source.

### Include ordering inside a `.cpp`

Own private → own public → other modules' public → std:

```cpp
#include "z.h"               // own private header        — "" + relative path
#include "inc/mod1/pub_a.h"  // own public header          — "" + relative path
#include <mod2/pub_q.h>      // another module's public    — <> + module prefix
#include <string>            // std                        — <>
```

## Recommended structure

Any layout that obeys the six rules is valid. The recommended shape:

```
CMakeLists.txt            ← top level: adds deps/ and src/ subdirectories
  deps/
    CMakeLists.txt        ← adds all libs, sets per-lib options
      lib1/ ...
      lib2/ ...
  src/
    CMakeLists.txt        ← adds all module + executable subdirectories
    exe1/                 ← executable: only private headers, NO inc/ dir
      CMakeLists.txt      ← links deps + modules, but NOT other executables
      bar.cpp             ← private implementation
      bar.h               ← private header
      main.cpp            ← entry point
    exe2/ ...             ← another executable (one CMake target per dir)
    mod1/                 ← a module (library)
      CMakeLists.txt      ← defines mod1::mod1; may link deps + other modules
      inc/                ← public headers (keep the name short: "inc")
        mod1/             ← subdir named after the module → part of the #include
          pub_a.h         ← public header; must NOT include own private headers
          pub_b.h         ← may include own public headers by relative "" path
      x.cpp               ← private implementation
      x.h                 ← private header
      z.cpp               ← (see include-ordering example above)
    mod1/ ...             ← another module (one CMake target per dir)
  tests/                  ← pytest + Catch2/CTest (see tests/AGENTS.md)
    cpp/ ...              ← Catch2 executables; links src/ libs + deps
    python/ ...           ← pytest suites; one pyproject.toml per leaf
```

## This repo's concrete conventions

Real modules in `src/` follow the rules above with a `cpp/` layer and a
namespaced alias. A header-only / interface module looks like:

```cmake
add_library(deviceio_base INTERFACE)

target_include_directories(deviceio_base
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>   # exports inc/, so consumers write <deviceio_base/...>
)

target_link_libraries(deviceio_base
    INTERFACE
        isaacteleop_schema
)

add_library(deviceio::deviceio_base ALIAS deviceio_base)     # mod::mod alias
```

So the on-disk path is `src/core/<module>/cpp/{CMakeLists.txt, inc/<module>/*.hpp, *.cpp, *.h}`,
and `src/core/<module>/CMakeLists.txt` simply `add_subdirectory(cpp)`. Match the
neighbouring modules' style (alias namespace, `$<BUILD_INTERFACE:...>`, INTERFACE
vs compiled library) rather than introducing a new pattern.

## Review checklist

When reviewing a diff that touches CMake, headers, or layout, flag any of:

- [ ] More than one target defined in a single leaf directory.
- [ ] Any `../` in an `#include` or in a CMake path.
- [ ] A target adding its own `inc/` or `.` to its include paths (breaks rule 3).
- [ ] Own headers included via `<...>` instead of relative `"..."`.
- [ ] Another module's / std headers included via `"..."` instead of `<...>`.
- [ ] Public include missing the module-name prefix (e.g. `<core.h>` not `<foo/core.h>`).
- [ ] Exported include dir using `PUBLIC`/`PRIVATE` instead of `INTERFACE`.
- [ ] A public header (`inc/...`) that includes the module's private headers.
- [ ] A never-exported (private) header placed under `inc/` instead of beside its `.cpp` (breaks rule 6).
- [ ] A non-library leaf module (executable / loadable plugin) that has an `inc/` directory at all (breaks rule 6).
- [ ] An executable that links against another executable.
- [ ] Include block not ordered: own private → own public → other public → std.
- [ ] New module that omits the `mod::mod` alias or diverges from neighbour style.

## Related repo policy

This complements the repo's `AGENTS.md` files: before editing/creating files
under a directory, read every `AGENTS.md` on the path up to the repo root, and
run `SKIP=check-copyright-year pre-commit run --all-files` before treating work
as done. New build files need the SPDX header used elsewhere in the repo.
