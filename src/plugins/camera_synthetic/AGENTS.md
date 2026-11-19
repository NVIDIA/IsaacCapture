<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `camera_synthetic` plugin

**CRITICAL (non-optional):** Before editing this package, complete the mandatory **`AGENTS.md` preflight** in [`../../../AGENTS.md`](../../../AGENTS.md) (read every applicable `AGENTS.md` on your paths, not just this file).

The rules every camera plugin follows are under "Publishing cameras" in [`../../core/camera/AGENTS.md`](../../core/camera/AGENTS.md). Read them first; what follows is this plugin's own.

- **This plugin is the template.** A vendor's camera plugin is made by copying it ([Writing a camera plugin](../../core/camera/README.md#writing-a-camera-plugin)), so keep it the smallest complete camera plugin: a backend in `core/`, a `main` of a few lines in `app/`, a `plugin.yaml`. What every camera plugin needs goes into the camera library, not here.
- **No SDK and no library beyond the repository's own, ever.** It is the one camera plugin that is built whenever plugins are (`BUILD_PLUGINS`), the tests of the camera interface run it, and the Python wheel bundles it under `isaaccapture/plugins/camera_synthetic`, where a plugin's binary must not need a shared library the wheel does not ship ([`src/core/AGENTS.md`](../../core/AGENTS.md)). It has no build option of its own for the same reason. `memory=gpu` goes through the camera library's `cuda_pool.hpp`, which loads the NVIDIA driver's library only when CUDA memory is asked for: the binary must keep starting where there is none.
- **The pattern is an interface.** `synthetic_fill` and `synthetic_frame_number` are what `camera_bench --check-synthetic` and the tests check frames against; change the pattern and all of them in one pass.
- **`camera_bench` reads any camera plugin.** It lives here because it checks the synthetic pattern; it links the camera library and this plugin's `core/`, never a camera SDK.
