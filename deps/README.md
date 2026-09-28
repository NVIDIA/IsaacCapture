<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Isaac Teleop Dependencies

This directory contains all project dependencies, organized by type.

## Structure

```
deps/
├── CMakeLists.txt           # Main dependencies build configuration
├── README.md                # This file
├── third_party/             # Third-party open source dependencies
│   ├── CMakeLists.txt       # Third-party dependencies build configuration (uses FetchContent)
│   └── README.md            # Third-party dependencies documentation
└── cloudxr/                 # CloudXR related files
```

## How It Works

The `deps/third_party/CMakeLists.txt` centrally manages all third-party dependencies using CMake FetchContent:

1. **OpenXR SDK**: Automatically fetched and built as a static library
2. **yaml-cpp**: Automatically fetched and built as a static library
3. **pybind11**: Fetched only when `BUILD_PYTHON_BINDINGS=ON` (header-only)
   - Makes `pybind11::module` target available to all Python binding modules

Dependencies are automatically downloaded during CMake configuration. No manual initialization is required.

This centralized approach prevents duplicate includes and ensures consistent configuration across all modules.

## CloudXR Dependencies

The `cloudxr/` folder contains files and configurations for deploying the NVIDIA CloudXR, it
uses Docker Compose to setup the entire system with 3 different containers:
1. CloudXR runtime
2. Web server that hosts the CloudXR Web XR app
3. WebSocket SSL proxy

## System Dependencies

These are not fetched by CMake — they must be installed on the build
machine and are located via `find_package`.

### Vulkan SDK / loader
- **Locator**: `find_package(Vulkan REQUIRED)`
- **Required by**: `viz/core/` when `BUILD_VIZ=ON`.
- **Linux**: `apt-get install libvulkan-dev` (provides headers + `libvulkan.so.1`).
- **Windows**: install the LunarG Vulkan SDK and ensure `VULKAN_SDK` env var is set.
  CI uses `humbletim/install-vulkan-sdk@v1.2`.
- **Min version**: 1.2 (Televiz checks `VK_API_VERSION_1_2` at device select time).
- **License**: Apache 2.0 (loader); per-vendor for ICD drivers.

### CUDA Toolkit
- **Locator**: `find_package(CUDAToolkit REQUIRED)`
- **Required by**: `viz/core/` (`CudaTexture`, `DeviceImage` link to
  `CUDA::cudart`) when `BUILD_VIZ=ON`.
- **Min version**: 12.0
- **CI**: installed via `Jimver/cuda-toolkit@v0.2.x` action with
  `nvcc` + `cudart` sub-packages.
- **License**: NVIDIA EULA

### glslangValidator (shader compiler)
- **Locator**: `find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)`
- **Required by**: `viz/shaders/` to compile `.vert` / `.frag` GLSL
  into SPIR-V at build time.
- **Linux**: `apt-get install glslang-tools`
- **Windows**: ships with the LunarG Vulkan SDK
- **macOS**: `brew install glslang`
- **License**: BSD-3 / Khronos

## Third-Party Dependencies

### OpenXR SDK
- **Source**: https://github.com/KhronosGroup/OpenXR-SDK.git
- **Version**: 75c53b6e853dc12c7b3c771edc9c9c841b15faaa (release-1.1.53)
- **Purpose**: OpenXR loader and headers for XR runtime interaction
- **Build**: Static library to avoid runtime dependencies
- **License**: Apache 2.0

### yaml-cpp
- **Source**: https://github.com/jbeder/yaml-cpp.git
- **Version**: f7320141120f720aecc4c32be25586e7da9eb978 (0.8.0)
- **Purpose**: YAML parsing for plugin configuration files
- **Build**: Static library to avoid runtime dependencies
- **License**: MIT

### pybind11
- **Source**: https://github.com/pybind/pybind11.git
- **Version**: a2e59f0e7065404b44dfe92a28aca47ba1378dc4 (v2.11.0-182)
- **Purpose**: C++/Python bindings for Isaac Teleop Python API
- **Build**: Header-only library
- **License**: BSD-style

### GLM
- **Source**: https://github.com/g-truc/glm.git
- **Version**: 1.0.1
- **Purpose**: Vulkan/GLSL math (`vec`, `mat`, `quat`, slerp / lookAt /
  perspective) used by the `viz` module for view & projection composition,
  pose math, and world/head/lazy-locked layer placement. Fetched only when
  `BUILD_VIZ=ON`.
- **Build**: Header-only library
- **License**: MIT

### spdlog
- **Source**: https://github.com/gabime/spdlog.git
- **Version**: v1.17.0
- **Purpose**: C++ logging backing `isaaccapture::Logger` (`src/core/log_bridge`)
  — named loggers, console/rotating-file sinks for standalone executables, and
  the socket sink that forwards C++ records into the Python logging session.
  Fetched unconditionally (not gated behind `BUILD_PYTHON_BINDINGS`), since
  standalone plugin executables need it too.
- **Build**: Static library
- **License**: MIT

## Adding New Dependencies

When adding new third-party dependencies:

To add a new third-party dependency:

1. Update `CMakeLists.txt` in this directory to add a FetchContent declaration:
   ```cmake
   FetchContent_Declare(
       <name>
       GIT_REPOSITORY <repository-url>
       GIT_TAG        <commit-sha-or-tag>
   )
   FetchContent_MakeAvailable(<name>)
   ```

2. Document it in this README with purpose, license, version, and repository information

3. If you need to preserve a specific version, use the full commit SHA in GIT_TAG

4. Update the [Build from Source](../docs/source/getting_started/build_from_source/index.rst) doc with any new requirements
