---
name: isaac-capture-install
description: >-
  Use when installing Isaac Capture (the isaaccapture Python package, formerly
  isaacteleop) into a project: creating a uv virtual environment, choosing extras
  (cloudxr, retargeters, retargeters-lite, ui, grounding, wuji, ovxr, desktop-mirror),
  installing a stable or pre-release build from PyPI, and checking that the install
  imports.
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

The documentation linked here tracks `main`, which can be ahead of the newest stable
release on PyPI: a class, function or extra it describes may be missing from a stable
install. When the project needs one of those, install a pre-release (step 3).

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
- `ovxr`: XR presentation through OVXR, for applications that render to a headset.
  See [OVXR installation](https://nvidia.github.io/IsaacCapture/main/getting_started/ovxr.md).
- `desktop-mirror`: the OVUI desktop mirror, used together with `ovxr`. It needs
  Python 3.12 or newer; on 3.11 it skips OVUI, so the mirror is unavailable.

Older releases lack `ovxr` and `desktop-mirror`. On a release without them, pip and
uv only warn that the extra does not exist and install the rest.

The Quick Start installs `cloudxr,retargeters`, which suits teleoperation with an XR
headset.

## 3. Install

The latest stable 1.x release:

```bash
uv pip install 'isaaccapture[cloudxr,retargeters]~=1.0' --upgrade-package isaaccapture
```

uv keeps an installed version that already satisfies the specifier, so
`--upgrade-package isaaccapture` is what replaces an older install in the environment.

Pre-releases are published on the NVIDIA index. The `a0` in the specifier opts in
`isaaccapture` alone; do not add `--prerelease=allow`, which also takes pre-releases
of every dependency:

```bash
uv pip install 'isaaccapture[cloudxr,retargeters]~=1.0a0' \
      --extra-index-url https://pypi.nvidia.com --upgrade-package isaaccapture
```

To match one version of the documentation exactly, use the pin its Quick Start shows.

## 4. Check the install

Run this from outside any Isaac Capture checkout:

```bash
python -c "import isaaccapture; print(isaaccapture.__version__, isaaccapture.__file__)"
```

Success prints the installed version number and a path inside the virtual
environment's `site-packages`. Anything else is a failed install:

- A path outside `site-packages`: the import found another `isaaccapture` tree,
  usually through `PYTHONPATH` or the current directory, and the version printed may
  be the installed one rather than that tree's. Unset `PYTHONPATH` or change
  directory, then check again.
- `0+unknown`: the same, with a tree that has no installed distribution at all.
- An `ImportError` saying the legacy `isaacteleop` distribution is still installed:
  rerun the install with `--upgrade`, keeping the same version and extras, to
  replace it.
- `ModuleNotFoundError: No module named 'isaaccapture'`: the virtual environment
  is not active, or the package went into a different environment.

`isaaccapture` installs an `isaacteleop` distribution with an epoch-1 version such as
`1!1.6.152` beside itself. That is a transition package with no code that replaces
the legacy one, so leave it installed; old `isaacteleop` imports are served from the
`isaaccapture` package.

Newer releases ship `llms.txt` and these skills inside the package. The
`llms.txt` beside the `__init__.py` printed above indexes the rest.

## References

- [Quick Start](https://nvidia.github.io/IsaacCapture/main/teleop/quick_start.md)
- [System requirements](https://nvidia.github.io/IsaacCapture/main/references/requirements.md)
- [Retargeting](https://nvidia.github.io/IsaacCapture/main/references/retargeting/index.md)
