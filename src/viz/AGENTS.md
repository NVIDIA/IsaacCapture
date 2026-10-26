<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `src/viz` (Televiz)

**CRITICAL (non-optional):** Before editing under `src/viz/`, complete the
mandatory **`AGENTS.md` preflight** in [`../../AGENTS.md`](../../AGENTS.md)
(read every applicable `AGENTS.md` on your paths, not just this file).

## Package shape

`src/viz/` mirrors **`src/core/`**: a peer container of sub-modules, not a
single sub-module. Each sub-module is its own static library. Tests live under
[`../../tests/`](../../tests/) — see [`tests/AGENTS.md`](../../tests/AGENTS.md).

- **`viz/core/`** — foundational types + Vulkan/CUDA infrastructure.
  Library: `viz_core`. Today: `VkContext`, `VizBuffer`, `Pose3D`,
  `Fov`, `Resolution`, `ViewInfo`, `PixelFormat`, `RenderTarget`,
  `FrameSync`, `HostImage`, `DeviceImage`. `HostImage` owns CPU bytes
  and exposes a `VizBuffer view()`; `DeviceImage` owns CUDA-Vulkan
  interop memory (VkImage + cudaArray_t) plus one timeline semaphore
  (`cuda_done_writing`) that layers expose to the compositor for
  fine-grained producer→consumer sync. The reverse direction (consumer
  done reading) is handled at the layer level by buffering enough
  slots that the producer never targets a still-in-flight read.
  Consumed via discrete accessors — no `view()` because `cudaArray_t`
  is opaque tiled memory, not a CUDA device pointer. `VizBuffer`
  covers linear pointer-backed memory (CPU bytes / CUDA device
  pointer; exposes `__cuda_array_interface__` / `__array_interface__`
  in Python). Math types (`glm::vec3`, `glm::quat`, `glm::mat4`) come
  from GLM 1.0.1 (FetchContent in `deps/third_party/`); use
  `glm::value_ptr(mat)` to get a raw `float*` for Vulkan / CUDA upload
  (POD-equivalent layout, no copy). CUDA-Vulkan interop requires CUDA
  Toolkit at link time (`CUDA::cudart_static` — statically linked, so
  consumers carry no runtime `libcudart` dependency). `VkContext::init()`
  matches the current CUDA device to the chosen Vulkan physical
  device by UUID — every viz_core type can assume CUDA and Vulkan
  are talking to the same GPU without re-doing the match.
- **`viz/layers/`** — the concrete layers. `LayerBase` itself lives in
  `viz/session/`. Library: `viz_layers` (STATIC). Depends PUBLIC on
  `viz::core` + `viz::session`, PRIVATE on `viz::shaders`. Today:
  `ImageLayerBase` plus `QuadLayer`, `CylinderLayer` and
  `EquirectLayer` on top of it, and `ProjectionLayer` directly on
  `LayerBase`. The three image layers are CUDA-fed via a mailbox of
  `DeviceImage`s owned by the shared `ImageLayerBase`, sized
  `kSlotCount = kMaxFramesInFlight + 2`; producer side is
  `submit(VizBuffer)` and the renderer always samples the most
  recently published slot.
  Pipelines built per-layer using the driver-side `VkPipelineCache`
  from `VkContext`. **Deferred:** a zero-copy `acquire`/`release`
  variant for producers that can write directly into a tiled
  `cudaArray_t` (NVDEC, custom CUDA kernels). The mailbox internals
  already track slot ownership, so the addition is local to
  `ImageLayerBase` and the layers built on it; revisit when a real
  producer demands it.
  Test-only fixture layers (`ClearRectLayer`, `ThrowingLayer`) live in
  `tests/cpp/viz/support/layers_testing/inc/viz/layers/testing/` and are exposed via
  the `viz::layers_testing` static library — used by `viz_session_tests`
  to compose into a `VizSession`.
- **`viz/session/`** — `VizSession`, `VizCompositor`, `FrameInfo`,
  `FrameTimingStats`, `SessionState`, `LayerBase`, `Swapchain`,
  `TileLayout`, and the display backends (offscreen, window/GLFW, and
  XR). Library: `viz_session`. Links `viz::core`, `viz::xr`,
  `oxr::oxr_utils`, `glfw`. Public API for the
  whole module — applications interact with Televiz through
  `VizSession::create()`.
- **`viz/xr/`** — OpenXR instance/session lifecycle, event polling, and
  the `xrWaitFrame`/`xrBeginFrame`/`xrEndFrame` loop. Swapchain wrapping
  and OpenXR↔`viz` type conversion live in `viz_session`'s `XrBackend`.
  Library: `viz_xr`. Always built under `BUILD_VIZ` — `viz_core` and
  `viz_session` both require the OpenXR loader. Depends on `viz_core` +
  OpenXR.
- **`viz/python/`** — pybind11 module `_viz`, exposed as `isaaccapture.viz`.
- **`viz/shaders/`** — GLSL → SPIR-V at build time. Library: `viz_shaders`
  (INTERFACE — exposes generated headers `viz/shaders/<name>.spv.h`,
  each containing an `inline constexpr alignas(uint32_t) unsigned char`
  byte array + a `Size` constant). Compilation runs `glslangValidator`
  (system-installed; CI gets `glslang-tools` apt package). Add new
  shader programs by dropping `<name>.vert` / `<name>.frag` in
  `viz/shaders/cpp/` and calling `compile_shader(<name>.vert kVarName)`
  in the local CMakeLists.

Catch2 and pytest suites for viz live under `tests/cpp/viz/<sub-module>/`
and `tests/python/viz/` respectively (one Catch2 executable per sub-module:
`viz_core_tests`, `viz_layers_tests`, …).

`src/viz/CMakeLists.txt` is an **orchestrator only** — it adds the
sub-module sub-directories. Sub-module `CMakeLists.txt` files build the
actual libraries.

The whole module is gated by **`BUILD_VIZ`** at the top level, which
defaults to `ON` when `find_package(Vulkan)` and `find_package(CUDAToolkit)`
both succeed and `OFF` otherwise — a machine that cannot build viz still
configures. Override with `-DBUILD_VIZ=ON/OFF`. Note this is *not* the
`BUILD_PLUGIN_OAK_CAMERA` convention, which is a hard `OFF`.
Build paths that ship viz (the wheel CI on Linux + Windows) pass
`-DBUILD_VIZ=ON` explicitly. Lean Dockerfiles
(`examples/teleop_ros2/Dockerfile`) get viz-free builds for free.

When `BUILD_VIZ=ON` the build machine must have:
- **Vulkan headers + loader**: `libvulkan-dev` on Linux, LunarG SDK on
  Windows.
- **CUDA Toolkit** (cudart for link; nvcc once we ship CUDA kernels):
  apt `nvidia-cuda-toolkit` or the official NVIDIA installer /
  CI action (`Jimver/cuda-toolkit`). The wheel excludes
  `libcuda.so.1` — consumers supply it via NVIDIA driver.
- **glslangValidator** for shader compilation: `glslang-tools` apt
  package on Linux, `brew install glslang` on macOS, ships with the
  Vulkan SDK on Windows.

## Code conventions

- **C++ namespace:** all Televiz symbols in `viz`. Internal helpers go in
  an anonymous namespace in the `.cpp` that needs them — there is no
  `viz::detail`. Test infrastructure in `viz::testing`.
  This is shared across all sub-modules — no per-module nested namespace.
- **Naming:** types `PascalCase`, methods/functions/variables
  `snake_case` (matches `src/core/`). Private members use trailing
  underscore (`instance_`). Enum values use `kPascalCase` (`kRGBA8`).
- **Include paths:** mirror the on-disk nesting since sub-modules are
  *children* of `viz/`, not peers of each other:
  `<viz/core/vk_context.hpp>`, `<viz/layers/quad_layer.hpp>`,
  `<viz/session/viz_session.hpp>`, `<viz/xr/openxr_session.hpp>`. Each
  sub-module's `inc/viz/<sub-module>/` lives under that sub-module's
  `cpp/`, so per-library isolation is preserved (linking only `viz::core`
  exposes only `viz/core/...` headers, not other sub-modules).
- **Library aliases:** drop the redundant `viz_` prefix in the alias —
  `viz::core`, `viz::layers`, `viz::session`, `viz::xr`. Real CMake
  target names use underscores (`viz_core`, `viz_layers`, ...) since
  `::` is reserved for ALIAS targets. Consumers always say
  `target_link_libraries(... PRIVATE viz::core)`, never the underscore
  form.
- **Format:** clang-format clean (Allman, 120 cols, 4-space indent, left
  pointer alignment). Enforced in CI. Run `clang-format-14 -i` locally
  on any file you modify.

## Tests

- C++ tests: **Catch2 v3**, `TEST_CASE("name", "[tag][tag]")`, linked
  against `Catch2::Catch2WithMain`.
- Tag conventions: **`[unit]`** (no GPU), **`[gpu]`** (Vulkan/CUDA
  required, must skip cleanly via `viz::testing::is_gpu_available`
  when no GPU), **`[xr]`** (OpenXR runtime required, manual-only).
- `catch_discover_tests(<target> ADD_TAGS_AS_LABELS)` — exposes Catch2
  tags as CTest labels. CI uses `ctest -L unit` for CPU selection; the GPU
  job runs each packaged `viz_*_tests` binary directly with the Catch2
  `[gpu]` filter, because it downloads bare executables with no build tree.
- GPU tests **must** use `GpuFixture` or call `is_gpu_available()` and
  `SKIP()` if false. Never assume a GPU is present.

### Required test patterns

Beyond happy-path coverage, every new feature MUST add tests for:

1. **Invalid input rejection** — assert public APIs throw on bad config
   (zero dimensions, uninitialized dependencies, unsupported modes)
   *before* any resource is allocated. Helps catch the
   "validate-then-allocate" failure mode.
2. **State-machine invariants** — for any class with implicit state
   (begin/end pairs, init/destroy lifecycles, lock/unlock), explicitly
   test every invalid transition: double-begin, end-without-begin,
   destroy-then-use, etc. Expect throw, no silent no-op (no-op masks
   real bugs in caller code).
3. **Exception recovery** — for any per-frame / per-iteration loop
   that includes user code (layer record(), callbacks), inject an
   exception via `viz::testing::ThrowingLayer` (or equivalent) and
   verify the *next* call still works. Catches fence-deadlock,
   leaked-flag, and partial-state bugs that only surface on retry.
4. **Idempotent destroy** — `destroy()` / cleanup methods must be
   safe to call twice (and after partial init failure). Test it.

### Test fixtures

Test-only `LayerBase` subclasses and helpers live in
`tests/cpp/viz/support/layers_testing/inc/viz/layers/testing/` and ship via the
`viz::layers_testing` static library. Other test executables link
that library to compose fixtures. Today:
- `ClearRectLayer` — paints a rect via `vkCmdClearAttachments` (no
  shaders); used to verify compositor dispatch produces real pixels.
- `ThrowingLayer` — throws from `record()` on a configurable schedule;
  used for exception-recovery tests.
- Catch2 sources live under `tests/cpp/viz/<sub-module>/`. One executable
  per sub-module (`viz_core_tests`, `viz_layers_tests`, …). Do **not**
  collapse viz tests into a single mega-target.

## CI coverage

- **`build-ubuntu`** (GitHub-hosted, GPU-less): builds `viz_*_tests`,
  runs `ctest --parallel` — `[unit]` tests pass, `[gpu]` tests SKIP
  cleanly. The job also packages all `viz_*_tests` binaries as the
  `viz-tests-${arch}` artifact.
- **`test-viz-gpu`** (self-hosted GPU runner, x64 + arm64): downloads
  the `viz-tests-${arch}` artifact and runs each binary with the
  `[gpu]` filter. This is where the GPU paths actually execute.
- **`publish-wheel`** requires `wheel-gate` to pass. Enabled GPU tests must
  succeed; `NO_GPU_TESTS=1` explicitly exempts the GPU jobs.

When you add a new `viz_<sub>_tests` executable, no CI changes needed:
the package step globs `viz_*_tests` and the runner loops over them.

## OpenXR boundary

- **Public API surface (in `viz_core`, `viz_layers`, `viz_session`)
  must not expose OpenXR types.** Convert `XrPosef`/`XrFovf` to
  `viz::Pose3D`/`Fov` at the boundary. The conversion lives in
  an anonymous namespace in `viz_session`'s `xr_backend.cpp`, where the
  OpenXR headers are already in scope.
- **Vulkan types are exposed** in the public API where functionally
  necessary (`VkCommandBuffer`, `VkRenderPass`, `VkImage` for custom
  layer authoring). This is intentional — Vulkan is the contract for
  the extension mechanism.
- **HMD wait from Python is interruptible.** `VizSession.create` releases
  the GIL and `OpenXrSession::wait_for_system` calls an optional poll
  hook; the binding uses it for `PyErr_CheckSignals` and restores the
  previous hook on exit (nested `create()`). Install Python SIGINT/SIGTERM
  handlers *before* `create()`, and do not put Python.h into `viz_xr`.
  C++ callers leave the hook null.

## Coordinate system

OpenXR stage space conventions throughout: **right-handed, Y-up,
meters, radians**. Robotics/ROS users converting from TF frames or
other coordinate systems must do so at the application boundary —
Televiz does not bridge to TF.
