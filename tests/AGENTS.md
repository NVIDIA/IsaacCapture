<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `tests/`

**CRITICAL:** Complete the mandatory **`AGENTS.md` preflight** in
[`../AGENTS.md`](../AGENTS.md) before editing here.

## Layout

All pytest and Catch2/CTest suites live under this tree:

```text
tests/
  cpp/       ← Catch2 executables (one target per module leaf)
  python/    ← pytest suites (one pyproject.toml per leaf)
```

Organize by **product area**, mirroring `src/` and `examples/`:

| Area | C++ path | Python path |
|------|----------|-------------|
| core | `tests/cpp/core/<module>/` | `tests/python/core/<module>/` |
| viz | `tests/cpp/viz/<module>/` | `tests/python/viz/` |
| plugins | `tests/cpp/plugins/<plugin>/` | — |
| examples | — | `tests/python/examples/<example>/` |

Shared C++ fixtures (not executables) live under `tests/cpp/viz/support/`
(`viz::test_support`, `viz::layers_testing`).

## Naming invariants (CI depends on these)

- **Catch2 executable names** stay stable: `schema_tests`, `viz_core_tests`,
  `viz_layers_tests`, … CI globs `viz_*_tests` for GPU packaging.
- **CTest name prefixes** stay stable: `schema_*`, `retargeting_*`,
  `viz_python_*`, etc.
- **Catch2 tags** → CTest labels via `ADD_TAGS_AS_LABELS` (`unit`, `gpu`, `xr`).

## Python conventions

- Each leaf directory has its own `pyproject.toml` so `uv run` resolves deps
  locally (mujoco pins, asyncio, cupy extras, grounding/wuji extras).
- After updating CloudXR import or mock-patch paths for the `isaaccapture`
  package, run `ruff format` on the touched tests; longer paths can change
  multiline layout.
- CTest runs one `pytest` invocation per `test_*.py` with `WORKING_DIRECTORY`
  set to that leaf.
- Use [`repo_paths.py`](python/repo_paths.py) for paths into `src/python/` or
  `examples/` from `conftest.py` — do not hard-code `parents[N]` against repo depth.
- If CMake under `examples/` reads a file under `tests/`, guard it on the
  in-tree configure: standalone wheel builds use the example as `CMAKE_SOURCE_DIR`,
  not the repo root.

## Adding tests

1. Pick the matching `tests/{cpp,python}/<area>/<module>/` leaf (create it if new).
2. Add sources + leaf `CMakeLists.txt` (copy an existing sibling).
3. Wire the leaf from the parent `CMakeLists.txt` under the same `BUILD_*` gates
   as before (e.g. viz tests require `BUILD_VIZ`).
4. Do **not** colocate new pytest/Catch2 trees under `src/` or `examples/`.

## Out of scope here

- `src/core/codegen/test_*.py` — unittest, co-located with the generator.
- `examples/oxr/python/test_*.py` — standalone scripts for CloudXR GPU CI.
- ROS2 Docker integration helpers under `examples/teleop_ros2/.../integration_tests/`.
