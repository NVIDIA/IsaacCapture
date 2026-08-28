<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OpenXR Tracking Examples

Examples demonstrating the modular OpenXR tracking API.

## Prerequisites

1. **Build the core library first:**
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
├── pyproject.toml          # the distribution
└── python/isaaccapture_examples/oxr/
    ├── modular_example.py
    ├── test_modular.py
    ├── test_extensions.py
    └── test_session_sharing.py
```

## Python Examples

### Running Python Examples

Install the example once, then run any module from anywhere:

```bash
uv pip install -e ./examples/oxr
export XR_RUNTIME_JSON=/path/to/cloudxr/openxr_cloudxr-dev.json
python -m isaaccapture_examples.oxr.test_modular
```

The installed tree works the same way — `uv pip install -e
install/examples/oxr` resolves against the wheel that build produced.

### Available Python Examples

- **modular_example** - Basic hand + head tracking
- **test_modular** - Complete API test
- **test_extensions** - Extension query demonstration
- **test_session_sharing** - Session sharing between DeviceIOSession instances

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
uv pip install -e install/examples/oxr
export XR_RUNTIME_JSON=/path/to/cloudxr/openxr_cloudxr-dev.json
python -m isaaccapture_examples.oxr.test_modular
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
