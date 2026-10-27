<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OpenXR Tracking Examples

Examples demonstrating the modular OpenXR tracking API.

## Prerequisites

1. **Build the core library first**, after the [source-build prerequisites](https://nvidia.github.io/IsaacCapture/main/development/build_from_source/index.html):
   ```bash
   # From project root
   cmake -B build
   cmake --build build
   cmake --install build
   ```
   This will build both the C++ static libraries and the Python wheels.

2. **For Python examples, install uv:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

## Directory Structure

```
examples/oxr/
├── cpp/                    # C++ examples
│   ├── CMakeLists.txt
│   ├── oxr_session_sharing.cpp
│   └── oxr_simple_api_demo.cpp
└── python/                 # Python examples
    ├── pyproject.toml      # uv configuration
    ├── modular_example.py
    ├── test_modular.py
    ├── test_extensions.py
    └── test_session_sharing.py
```

## Python Examples

Python examples use `uv` for dependency management and execution.

### Running Python Examples

After building and installing, navigate to the installed examples:

```bash
# From install directory (recommended)
cd install/examples/oxr/python
export XR_RUNTIME_JSON=/path/to/cloudxr/openxr_cloudxr-dev.json
uv run test_modular.py
```

Or run from source directory:

```bash
# From source directory (requires build to be complete)
cd examples/oxr/python
export XR_RUNTIME_JSON=/path/to/cloudxr/openxr_cloudxr-dev.json
uv run test_modular.py
```

### Available Python Examples

- **modular_example.py** - Basic hand + head tracking
- **test_modular.py** - Complete API test
- **test_extensions.py** - Extension query demonstration
- **test_session_sharing.py** - Session sharing between DeviceIOSession instances

## C++ Examples

C++ examples are built with CMake and linked against the static libraries.

### Building C++ Examples

From the top-level project directory:
```bash
cmake -B build
cmake --build build
cmake --install build
```

### Running C++ Examples

**oxr_session_sharing** - Demonstrates session sharing between multiple DeviceIOSession instances
**oxr_simple_api_demo** - Demonstrates the simple API

```bash
export XR_RUNTIME_JSON=/path/to/cloudxr/openxr_cloudxr-dev.json

# From build directory
./build/examples/oxr/cpp/oxr_session_sharing
./build/examples/oxr/cpp/oxr_simple_api_demo

# Or from install directory
./install/examples/oxr/cpp/oxr_session_sharing
./install/examples/oxr/cpp/oxr_simple_api_demo
```

## Quick Test

### Python Example
```bash
# From project root
cmake -B build
cmake --build build
cmake --install build

# Run Python example
cd install/examples/oxr/python
export XR_RUNTIME_JSON=/path/to/cloudxr/openxr_cloudxr-dev.json
uv run test_modular.py
```

### C++ Example
```bash
# From project root
cmake -B build
cmake --build build

# Run C++ example
export XR_RUNTIME_JSON=/path/to/cloudxr/openxr_cloudxr-dev.json
./build/examples/oxr/cpp/oxr_session_sharing
```

## Build Outputs

After building:
- **C++ Static Libraries**: Built in `build/src/core/`
- **Python Wheel**: `build/wheels/isaaccapture-*.whl`
- **C++ Examples**: `build/examples/oxr/cpp/`

## Documentation

See `../../src/core/` for module documentation.
