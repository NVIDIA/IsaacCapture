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

The `test_*.py` scripts in [`tests/python/xr_runtime/`](../../tests/python/xr_runtime/)
exercise the same API. CloudXR GPU CI runs the ones listed in
[`scripts/run_tests_with_cloudxr.sh`](../../scripts/run_tests_with_cloudxr.sh).
To run one by hand, use `python tests/python/xr_runtime/<script>.py` from the
project root, sourcing the CloudXR environment first as for the C++ examples.
The scripts that start a plugin look in `install/plugins` after
`cmake --install build`, or pass `--plugin-root`.

## Python Examples

### Running Python Examples

Install `isaaccapture` from this build into a virtual environment made with the
build's Python and activated, as in
[Build from source, step 4](https://nvidia.github.io/IsaacCapture/main/development/build_from_source/index.html#install-the-isaaccapture-pip-package).
Then, from the project root, with `X.Y` the first two numbers of `VERSION`:

```bash
uv pip install "isaaccapture[cloudxr]==X.Y+local" --find-links=./install/wheels/ --reinstall-package isaaccapture
uv pip install -e ./examples/oxr
python -m isaaccapture_examples.oxr.modular_example
python -m isaaccapture_examples.oxr.modular_example_with_mcap   # writes recordings/tracking_recording_<timestamp>.mcap
```

Each module attaches to the running CloudXR runtime, or starts one in the
background and leaves it running; stop it with
`python -m isaaccapture.cloudxr.service stop`. The first start asks you to
accept the CloudXR EULA, and `--accept-eula` accepts it without the prompt. To
use a different OpenXR runtime, pass `--no-launch-cloudxr-runtime` as in
[Use a system OpenXR runtime](https://nvidia.github.io/IsaacCapture/main/references/cloudxr.html#use-a-system-openxr-runtime-no-cloudxr-attach).

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

They do not start CloudXR. With the service running
(`python -m isaaccapture.cloudxr.service start --accept-eula`), source its
environment, which sets `XR_RUNTIME_JSON` and the `NV_CXR_*` variables:

```bash
source ~/.cloudxr/run/cloudxr.env

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

# Run Python example. Replace 3.11 with your ISAAC_TELEOP_PYTHON_VERSION and
# X.Y with the first two numbers of VERSION.
uv venv --python 3.11 --seed .venv
source .venv/bin/activate
uv pip install "isaaccapture[cloudxr]==X.Y+local" --find-links=./install/wheels/ --reinstall-package isaaccapture
uv pip install -e ./examples/oxr
python -m isaaccapture_examples.oxr.modular_example
```

### C++ Example
```bash
# From project root
cmake -B build
cmake --build build

# Run C++ example, with the CloudXR service running
source ~/.cloudxr/run/cloudxr.env
./build/examples/oxr/cpp/oxr_session_sharing
```

## Build Outputs

After building:
- **C++ Static Libraries**: Built in `build/src/core/`
- **Python Wheel**: `build/wheels/isaaccapture-*.whl`
- **C++ Examples**: `build/examples/oxr/cpp/`

## Documentation

See `../../src/core/` for module documentation.
