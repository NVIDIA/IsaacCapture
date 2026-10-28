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
├── pyproject.toml          # the distribution
└── python/isaaccapture_examples/oxr/
    ├── __init__.py
    ├── modular_example.py
    └── modular_example_with_mcap.py
```

The `test_*.py` scripts that CloudXR GPU CI runs against this API are in
[`tests/python/xr_runtime/`](../../tests/python/xr_runtime/).

## Python Examples

### Running Python Examples

Install `isaaccapture` built from this tree into the active environment
first, for example
`uv pip install "isaaccapture[cloudxr]" --find-links=./install/wheels/`.
Then install the example once and run a module from anywhere:

```bash
uv pip install -e ./examples/oxr
export XR_RUNTIME_JSON=/path/to/cloudxr/openxr_cloudxr-dev.json
python -m isaaccapture_examples.oxr.modular_example
```

### Available Python Examples

- **modular_example** - Basic hand + head tracking
- **modular_example_with_mcap** - The same trackers recorded to an MCAP file, then replayed

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
uv pip install "isaaccapture[cloudxr]" --find-links=./install/wheels/
uv pip install -e ./examples/oxr
export XR_RUNTIME_JSON=/path/to/cloudxr/openxr_cloudxr-dev.json
python -m isaaccapture_examples.oxr.modular_example
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
