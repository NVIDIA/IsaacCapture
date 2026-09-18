<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sharpa Avatar visualization example

Live OpenXR hands from Sharpa Avatar gloves in the browser, plus pinch haptic.
CloudXR and `avatar_hand_plugin` start with the example unless you opt out.

```text
gloves -> avatar_hand_plugin -> TeleopSession -> visualization and haptics
```

The C++ plugin and Avatar SDK stay under
[`src/plugins/sharpa_avatar`](../../src/plugins/sharpa_avatar/README.md).
Install them first, then run this example against the installed plugin.

## Run

```bash
./src/plugins/sharpa_avatar/install.sh
uv pip install -e ./examples/sharpa_avatar
python -m isaacteleop_examples.sharpa_avatar
```

Open the URL it prints (default <http://127.0.0.1:8080>). The left hand is cyan
and the right hand is orange. The terminal reports the OpenXR hands plus the
22-DoF RAW and ROBOT joint streams. Bringing a fingertip close to the thumb
vibrates that finger. Ctrl+C stops it.

If the plugin binary is missing, run the installer above first. `uv pip install
-e` pulls `viser`; CloudXR comes from the `isaacteleop[cloudxr]` extra.

## Options

```text
--no-haptic                   disable pinch feedback
--no-viz                      terminal + haptic only (no browser view)
--host / --port               viser bind (default 127.0.0.1:8080)
--no-launch-plugin            connect to an already running plugin
--no-launch-cloudxr-runtime   connect to an already running CloudXR runtime
--world-frame                 display the unmodified OpenXR poses
--plugin-search-path DIR      directory that contains sharpa_avatar/
```

## Behavior

`live_view.py` launches the plugin through `TeleopSession`, reads `/hand/left`
and `/hand/right` plus the RAW/ROBOT joint collections, and draws both
skeletons in a stable local frame (the gloves have no world-space wrist). Pass
`--world-frame` to skip that placement.

Pinch haptic uses the same per-finger `HapticSink` path as the generic glove
example, targeted at collection `avatar_glove_haptic`.
