<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OpenXR Python Examples

Python examples demonstrating the OpenXR tracking API using the `isaaccapture` Python wheel.

## Prerequisites

1. **Build and install the project** from the project root, after the
   [source-build prerequisites](https://nvidia.github.io/IsaacCapture/main/development/build_from_source/index.html):
   ```bash
   cd ../../..
   cmake -B build
   cmake --build build
   cmake --install build
   ```
   This creates the Python wheel in `build/wheels/` and installs it to `install/wheels/`.

2. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

## Running Examples

### From Install Directory (Recommended)

After building and installing, run examples from the install directory:

```bash
cd ../../../install/examples/oxr/python
export XR_RUNTIME_JSON=/path/to/your/openxr_runtime.json
uv run test_modular.py
```

### From Source Directory

You can also run from the source directory:

```bash
export XR_RUNTIME_JSON=/path/to/your/openxr_runtime.json
uv run test_modular.py
```

The `pyproject.toml` configuration automatically locates the built wheel.

## Available Examples

### modular_example.py
Basic demonstration of hand and head tracking.

```bash
uv run modular_example.py
```

### test_modular.py
Comprehensive test of the modular tracking API.

```bash
uv run test_modular.py
```

### test_extensions.py
Demonstrates querying available OpenXR extensions.

```bash
uv run test_extensions.py
```

### test_session_sharing.py
Shows how to share sessions between multiple DeviceIOSession instances.

```bash
uv run test_session_sharing.py
```

## Troubleshooting

### Wheel not found
If you get an error about the wheel not being found, build the project:
```bash
cd ../../..
cmake -B build
cmake --build build
```

### uv not installed
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Then restart your shell
```

### OpenXR runtime errors
Set the XR_RUNTIME_JSON environment variable:
```bash
export XR_RUNTIME_JSON=/path/to/your/openxr_runtime.json
```
