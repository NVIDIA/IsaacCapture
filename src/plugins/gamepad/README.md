<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Gamepad Plugin

Reads a gamepad through [SDL3](https://github.com/libsdl-org/SDL) and pushes `GamepadSnapshot`
via OpenXR. The inputs are `GamepadInputState` from `gamepad.fbs`: the buttons
and axes every standard pad has, the same set as OpenXR's `xbox_controller` profile, as named fields
like `ControllerInputState`'s, with face buttons named by position (`south_click`, `west_click`,
...). SDL resolves each pad's own numbering through its drivers and
[SDL_GameControllerDB](https://github.com/mdqinc/SDL_GameControllerDB) mappings, so `west_click` is X
on an Xbox pad and Square on a PlayStation pad. Read it with `GamepadTracker`, `GamepadSource`, or
any reader using the same `collection_id`. Linux only.

The plugin is also bundled in the `isaaccapture` package: a `TeleopSession` starts it with
`PluginConfig(plugin_name="gamepad", plugin_root_id="gamepad", search_paths=[BUNDLED_PLUGINS_DIR])`.

Supported pads: Xbox (360, One, Series, USB and Bluetooth), PlayStation (DualShock 4, DualSense),
Nintendo Switch Pro, Steam Controller, and any pad with an SDL_GameControllerDB mapping. Validated:
Xbox 360 Controller.

SDL3 (zlib license) is fetched at configure time and linked statically with only its joystick and
HIDAPI subsystems, so the binary needs nothing beyond libc/libstdc++.

## Device access

On a desktop session udev grants the logged-in user access to gamepad nodes, so no setup is
needed. Without a local seat (SSH, a service, a container) add the user to the `input` group, and
pass the pad's `/dev/input/event*` node into a container.

## Usage

```bash
./gamepad_plugin [--player=N] [--collection-id=ID]
```

- **--player**: Read the Nth connected gamepad, in connection order (default 1). The pad is kept
  until it disconnects; for several pads, run one plugin per player with its own collection.
- **--collection-id**: Default `gamepad`. Match this when creating `GamepadTracker`.
- **--plugin-root-id**: Set by the plugin launcher; enables device status reporting.

The plugin pushes at 90 Hz. While no pad is open it keeps pushing snapshots without inputs, and
nothing held, so a reader never holds the last stick deflection of a pad that went away. The pad's
name, type and USB IDs are logged when it is opened.

A pad without a mapping is not listed as a gamepad. Give SDL one through the
`SDL_GAMECONTROLLERCONFIG` environment variable (an SDL_GameControllerDB line) in the plugin's
environment.

## Data

- `inputs`: `GamepadInputState`. Sticks are in `[-1, 1]` with right and up positive, triggers in
  `[0, 1]`.

## Device status

With `--plugin-root-id`, the plugin reports device `/gamepad` as `CONNECTED`, `DISCONNECTED` (no
pad for this player), or `FAILED` (SDL found the pad but could not open it).
