---
name: isaac-capture-install
description: >-
  Use when installing Isaac Capture (the isaaccapture Python package, formerly
  isaacteleop) into a project: creating a uv virtual environment, choosing extras
  (cloudxr, retargeters, retargeters-lite, ui, grounding, wuji), installing a stable
  or pre-release build from PyPI, and checking that the install imports.
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Install Isaac Capture

Install the `isaaccapture` package from PyPI into a virtual environment, then check
that it imports. The
[Quick Start](https://nvidia.github.io/IsaacCapture/main/teleop/quick_start.md) is the
source of truth for these commands; follow it wherever it disagrees with this skill.

## 1. Create a virtual environment

Python 3.11 to 3.13 are supported. Install into a virtual environment: outside one,
`pip install` fails on Ubuntu 24.04 with `externally-managed-environment` and
`uv pip install` fails with `No virtual environment found`.

```bash
uv venv --python 3.11 --seed .venv
source .venv/bin/activate
```

`--seed` adds pip, so plain `pip install` also works in the venv. To use an existing
venv, or a conda environment other than `base`, activate it and skip `uv venv`.
Activate the environment again in every new terminal.

## 2. Choose extras

Extras add optional dependencies. Pick the ones the project needs:

- `cloudxr`: streaming to an XR headset through the CloudXR runtime. The teleop
  examples and camera streaming need it.
- `retargeters`: the full retargeting stack (dex-retargeting, PyTorch, Pinocchio),
  for dexterous-hand and joint-space retargeters.
- `retargeters-lite`: SciPy only, enough for lightweight retargeters such as
  `Se3AbsRetargeter`.
- `ui`: the imgui parameter-tuning UI.
- `grounding`: the runtime stack of the Sharpa hand retargeter.
- `wuji`: the Wuji glove retargeter.

The Quick Start installs `cloudxr,retargeters`, which suits teleoperation with an XR
headset.

## 3. Install

The latest stable 1.x release:

```bash
uv pip install 'isaaccapture[cloudxr,retargeters]~=1.0'
```

Pre-releases are published on the NVIDIA index and need an opt-in:

```bash
uv pip install 'isaaccapture[cloudxr,retargeters]~=1.0' \
      --extra-index-url https://pypi.nvidia.com --prerelease=allow
```

To match one version of the documentation exactly, use the pin its Quick Start shows.

## 4. Check the install

```bash
python -c "import isaaccapture; print(isaaccapture.__version__)"
```

Success prints the installed version number. Anything else is a failed install:

- `0+unknown`: the import found an `isaaccapture` tree with no installed
  distribution, such as a build directory on `PYTHONPATH`. Install the package and
  run from outside that tree.
- An `ImportError` saying the legacy `isaacteleop` distribution is still installed:
  rerun the install with `--upgrade`, keeping the same version and extras, to
  replace it.
- `ModuleNotFoundError: No module named 'isaaccapture'`: the virtual environment
  is not active, or the package went into a different environment.

## References

- [Quick Start](https://nvidia.github.io/IsaacCapture/main/teleop/quick_start.md)
- [System requirements](https://nvidia.github.io/IsaacCapture/main/references/requirements.md)
- [Retargeting](https://nvidia.github.io/IsaacCapture/main/references/retargeting/index.md)
