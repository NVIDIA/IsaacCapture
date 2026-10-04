<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Keyboard Plugin

Reads a keyboard from `/dev/input/event*` and pushes `KeyboardOutput` via OpenXR. Use with `KeyboardTracker` with the same `collection_id`.

Reports raw press state only, with no semantic mapping to axes or commands -- that mapping belongs
in a retargeter (e.g. `KeyboardToSe3RelRetargeter`) consuming this tracker's output.

Self-discovers its device (the first `*-event-kbd` entry under `/dev/input/by-path/`), so it needs
no arguments to run and can be auto-launched by `PluginManager` via `PluginConfig` -- no manual
process to start.

## Usage

Auto-launched (recommended -- matches how `PluginManager` invokes plugins):

```bash
./keyboard_plugin --plugin-root-id=keyboard
```

Manual / standalone, with an explicit device:

```bash
./keyboard_plugin [device_path] [--plugin-root-id=<collection_id>]
```

- **device_path**: Optional. Defaults to the first `*-event-kbd` entry under `/dev/input/by-path/`.
  Identify a specific keyboard event device with `cat /proc/bus/input/devices` (look for a
  `Handlers=... eventN` line under a keyboard entry) or `evtest`. Reading `/dev/input/event*`
  typically requires membership in the `input` group.
- **collection_id**: Default `keyboard`. Match this when creating `KeyboardTracker`.

## Key mapping

Reports the current press state of: `W`, `A`, `S`, `D`, `Q`, `E`, `Z`, `X`, `T`, `G`, `C`, `V`, `K`.
Autorepeat events are ignored (press state is already true).

Linux only.
