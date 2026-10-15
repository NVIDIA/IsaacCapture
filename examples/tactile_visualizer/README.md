<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# TactileShape visualizer and editor

A local developer tool for creating, inspecting, editing, and exporting
`TactileShape`, while optionally showing Sharpa glove URDF forward kinematics,
OpenXR hand joints, and `TactileData` forces. The browser is a view onto one
server-side model; the first connected browser owns the editing lease and later
clients are read-only.

```text
avatar_raw_{left,right} -> JointStateSource -> URDF update_cfg() -> glove FK
/hand/{left,right}      -> HandsSource      -> OpenXR skeleton --+
tactile_{left,right}    -> TactileTracker   -> shape + forces ----+-> viser
                                                             latest-only slot
```

The tool uses meters and XYZW quaternions in schema files. Position fields in
the UI are centimeters. `TactileShape` taxels are animated with the schema's
four-weight linear-blend-skinning equation. Force is in newtons and uses
`t = clamp(||force|| / 30, 0, 1)`, linearly interpolating blue to red.

Coordinate and unit conventions:

- Schema positions are hand-root-space meters; GUI positions are centimeters.
- Schema and hand-source quaternions are XYZW; Viser receives WXYZ.
- Joints use `XrHandJointEXT` ordinals 0–25. Live poses are transformed into
  wrist-root space; the wrist is world origin.
- The stored shape is canonical right-hand data. Left display reflects X for
  the skeleton, taxels, and URDF without changing exported bytes.
- A taxel frame uses tangent as X, normal as Z, and `normal × tangent` as Y;
  edits re-normalize and orthogonalize the two stored vectors.
- Raw glove values are passed unchanged to matching URDF actuated joints
  (normally radians). URDF config offset is centimeters, scale is unitless,
  and config rotation is XYZW.

## Install and run

```bash
# Python 3.11+ and an IsaacCapture build exposing its Python modules.
uv pip install -e ./examples/tactile_visualizer

# Safe offline editor (no OpenXR runtime or hardware).
python -m isaaccapture_examples.tactile_visualizer \
  --urdf /path/to/sharpa_hand.urdf \
  --urdf-config ./examples/tactile_visualizer/urdf_config.example.yaml \
  --shape /optional/existing_shape.bin

# Deterministic palm/fist animation, raw joints, shape, and force samples.
python -m isaaccapture_examples.tactile_visualizer \
  --urdf /path/to/sharpa_hand.urdf --mock

# Live DeviceIO. This starts CloudXR; launch the installed Sharpa plugin too.
python -m isaaccapture_examples.tactile_visualizer \
  --urdf /path/to/sharpa_hand.urdf --live --launch-plugin --accept-eula

# Headless MCAP replay; channel names are configurable.
python -m isaaccapture_examples.tactile_visualizer \
  --urdf /path/to/sharpa_hand.urdf --replay capture.mcap
```

The live path deliberately reuses the Sharpa sample's `HandsSource`,
`JointStateSource`, RAW joint names, collection IDs, `PluginConfig`, and
`TeleopSession`. Tactile input is added through `TactileTracker`; each hand's
collection is configurable with `--left-tactile-collection` and
`--right-tactile-collection`. MCAP channel labels are independently configurable
with the corresponding `--*-tactile-channel` options.

Use `urdf_config.example.yaml` to map Sharpa raw names to actuated URDF names and
to set the URDF scale/root transform. Unmapped URDF joints first try the same raw
name and otherwise receive zero. `ViserUrdf.update_cfg()` evaluates and renders
the URDF forward kinematics.

For a live session:

1. Build/install IsaacCapture, the Sharpa Avatar plugin, and its glove URDF.
2. Copy and adjust `urdf_config.example.yaml`, mapping actuated URDF joints to
   the documented `avatar_raw_{left,right}` joint names.
3. Run with `--live --launch-plugin --accept-eula`; add `--sdk-config`,
   `--plugin-search-path`, and tactile collection overrides when required.
4. Confirm that the toolbar reaches `LIVE_PLAYING`. This requires a skeleton
   sample, raw-joint sample, fixed producer shape, and length-matched tactile
   sample for the selected hand.

A tactile collection/session has one fixed metadata `TactileShape`; a producer
with a changed layout must create a new collection/session. The tool caches that
metadata and validates every force count against it. `--liveness-timeout`
defaults to 1 second. Producer and renderer communicate through a one-slot
latest-value mailbox, so input cannot create an unbounded queue.

## Interaction and states

- Click a taxel to select it. Shift-click a hand joint to inspect it.
- In `Box select`, an unmodified rectangle selects taxels and Shift-drag
  selects joints. Use `Navigate` for native orbit/pan/zoom. Clicking empty
  space clears the selection.
- `Translate` exposes translation axes. `Rotate` exposes rotation rings and
  changes taxel normal/tangent frames without moving taxel positions. Viser's
  transform control does not report mouse buttons, so the explicit mode toggle
  is the unambiguous supported interaction. Its native gizmo supplies axis hover
  highlighting.
- A single taxel uses its local frame; a multi-taxel gizmo uses world axes at
  the selection centroid. Joint axes are display-only.
- Live playback is read-only. Pause, disconnect, or use offline mode to edit.
  The producer keeps running while paused, but the 3D view remains frozen and
  only the newest unread sample is retained.
- Loading or editing creates a local shape override. Resume never silently
  replaces it with producer metadata. Switching hands mirrors the canonical
  shape and URDF for display without changing exported bytes.

The explicit states are `OFFLINE`, `LIVE_PLAYING`, `LIVE_PAUSED`, and
`DISCONNECTED`. Live-message time, not one-time metadata receipt, drives the
configurable timeout. The last view is preserved and becomes editable.

## Files and default shape

`Load TactileShape` validates the uploaded FlatBuffer before replacing the
active model. `Export active shape` downloads exact `TactileShape` bytes.
`Create New` downloads a 26-joint, zero-taxel shape but intentionally does not
replace the active model. Pass `--allow-server-write-root DIR` to additionally
enable an atomic local write constrained below that explicit directory.

`Add Taxels` appends 0–125 points in x-outer/y-middle/z-inner order from the
five-value grid on each axis. The schema has no unbound sentinel, so new taxels
repeat wrist index 1 in all slots and use weights `[1, 0, 0, 0]`.

OpenXR defines the 26 joint ordinals and runtime joint locations, but does not
define one universal bind pose. The built-in right-hand neutral pose is therefore
a clearly labeled tool default, not a device calibration. It follows the
[Khronos joint order](https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html#XrHandJointEXT)
and is suitable for bootstrapping files that should later be calibrated against
the intended glove.

The right-side schema hierarchy is collapsed by default. Shape leaves are
editable only offline or while paused; live data is read-only. Viser 1.1.1 has
no folder-expansion callback, so leaf controls exist server-side immediately
but remain visually hidden until their folders are opened.

## Validation boundary

Automated tests cover serialization, malformed input, state transitions,
latest-only behavior, skinning/edit transforms, projection selection, force
colors, configuration, and the mock source. CI cannot establish physical glove
accuracy, browser pointer ergonomics, or vendor URDF alignment; follow the
manual checklist in [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) on the
target setup before treating those aspects as verified.
