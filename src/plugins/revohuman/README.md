<!--
SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# RevoHuman plugin

The plugin reads 21 glove angles from the unmodified RevoHuman C SDK, takes the
SDK's hand points for them and maps those to 26 wrist-relative OpenXR joints with
the reviewed corrections in its `kinematics/` module; no URDF is loaded at
runtime. The `isaaccapture` wheel does not contain this plugin or the SDK; start
it from the CMake install tree.

Prerequisites, installation, running, the wrist pose and troubleshooting are
covered in the
[RevoHuman Glove](../../../docs/source/device/hand_tracking/revohuman.rst)
device page. This README adds the build details, the hand calculation and the
tactile contract.

## Tactile streams

| Stream | Contract |
| --- | --- |
| `<prefix>_<side>_tactile` / `tactile` | `TactileDataArray.taxel_forces`, taxel-local, the SDK's reading unchanged |
| `<prefix>_<side>_tactile/metadata` / `metadata` | bare `TactileMetadata`, LBS + HandSkeletonRef, 68,864-byte cap |

A taxel's segment index addresses `joint_ids`, not the OpenXR enum directly.
For example, with `joint_ids=[INDEX_TIP, WRIST]`, index 0 means INDEX_TIP and
index 1 means WRIST. Taxel pose orientation stores the tangent/normal frame.

## Build and install

Follow the repository's build-from-source prerequisites first: CMake >= 3.24,
a C++20 compiler, uv, Python 3.11–3.13, clang-format-14, and patchelf. Put the
selected Python environment and `~/.local/bin` on PATH.

`install.sh` gets the C SDK in one of two ways:

- By default it downloads the public release pinned at the top of the script,
  checks it against the pinned SHA-256 for the host architecture (x86_64 or
  aarch64) and extracts it next to the script, where later runs reuse it.
- `--sdk-root <prefix>` (or `REVOHUMAN_SDK_ROOT`) uses an extracted C SDK
  package: `include/revohuman.h`, `lib/librevohuman.so` with its versioned
  names, and `LICENSE`.

A Python SDK wheel cannot replace the C headers and library. The hand model
assumes the SDK's 21-angle order; an SDK that reorders the angles needs a review
of the plugin mapping, which the configure-time function checks do not catch.

Run from the IsaacCapture repository root:

```bash
CMAKE_BUILD_PARALLEL_LEVEL=4 ./src/plugins/revohuman/install.sh \
    --build-dir "$PWD/build-revohuman" \
    --install-prefix "$PWD/install" \
    -- -DBUILD_VIZ=OFF -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_EXAMPLES=OFF
```

Add `--sdk-root /path/to/revohuman-c-sdk` to use an SDK package you already have.

| Path | Contents |
| --- | --- |
| `revohuman-c-sdk-<version>-linux-<arch>/` (next to `install.sh`) | Downloaded release, reused on later runs; ignored by git |
| `<build-dir>` | Framework/plugin build; defaults to `<repo>/build` |
| `<install-prefix>/plugins/revohuman` | Plugin, SDK shared library and SDK licence (`LICENSE.revohuman-sdk-c`) |

Without `--install-prefix`, the build's `CMAKE_INSTALL_PREFIX` applies (a fresh
framework build defaults to `<repo>/install`).

The installed plugin finds the SDK library next to itself through `$ORIGIN`,
so it needs neither the SDK download nor its build directory. Keep the
library's symlinks. An OpenXR runtime and USB device permissions are still
required.

## Hand poses

Each glove's mechanical model follows the product the SDK identifies once the
glove is open, through the SDK's `rh_variant_hand_model`: P1 uses P and PTV1
uses DV1. There is no option to choose it.

Started by hand, the plugin publishes `<id>/device_status` only with
`--plugin-root-id=<id>`; consumers that gate on it show no forces otherwise.

For every glove sample the plugin passes the SDK angles to `rh_hand_points` for
the glove's model and uses its 27 points: 21 joint child links, five fingertips
and the palm_link origin, in the `{hand}_tracker_link` frame. Positions are meters
and rotations are row-major 3x3 matrices; the SDK writes NaN for every quantity
an invalid angle affects. The plugin re-expresses the points from WRIST (the
tracker origin becomes PALM, the palm_link origin moved to the depth of PALM
becomes WRIST), takes the SDK fingertips as given and applies the reviewed
OpenXR mapping below. It computes no FK of its own.

The SDK's P model has its own link positions and fingertips. The corrections
fitted on DV1 have been reviewed for P and apply to it unchanged. A P1 glove has
no tactile sensors: it streams the hand and joint states like a PTV1 glove, its
hand has no tactile collections, and its tactile status is disabled with "this
glove has no tactile sensors".

The plugin requires `rh_variant_hand_model` (C SDK 0.1.1 or later),
`rh_hand_model_info` and `rh_hand_points` with points version 1, all detected
during configuration. Without them the plugin stops before the OpenXR session
and names the reason, and `install.sh` refuses the install. With an SDK whose
library is missing or lacks a function the glove source calls, no plugin
executable is built at all. Each SDK model must match the SHA-256 reviewed for
it in `kinematics/openxr_model.inc`; for DV1 that is
`9e343e1f1ffccee7bf8dbfcd24bf05b085ff400668c9bc3b071fd174b7d55592`. A different
model is rejected before opening the gloves.

## Hand calculation boundary

The numeric geometry is the reviewed mapping in `kinematics/openxr_model.inc`:
for each hand, the WRIST rotation relative to palm_link and the per-point side
and height corrections. It was fitted on the SDK DV1 model and reviewed on the
P model, both named by their SHA-256 there. Neither compilation nor runtime reads
a URDF.

`live/live_source.cpp` reads the SDK angles and calls `rh_joints_to_radians`.
`kinematics/hand_kinematics.cpp` calls `rh_hand_points`, and
`kinematics/hand_geometry.cpp` maps the points to OpenXR order and applies the
corrections.

The SDK's 21 link points follow its angle order, **not** the OpenXR order:

| Finger | SDK angle/link slots | OpenXR output slots |
| --- | --- | --- |
| Index | 0–3 | 7, 8, 9, 10 |
| Middle | 4–7 | 12, 13, 14, 15 |
| Ring | 8–11 | 17, 18, 19, 20 |
| Little | 12–15 | 22, 23, 24, 25 |
| Thumb | 16–20 | 2, 3, 4, 5 |

The first four groups run from the distal link towards the palm. For the thumb,
slot 20 is the proximal anchor, 17 and 16 the next two points, and slot 19 sets
the proximal orientation. Output slot 0 is PALM, 1 is WRIST, and 6/11/16/21 are
invalid metacarpal placeholders. These six poses remain fixed relative to WRIST
when finger angles change. The C++ `kFingerMap` table states SDK and output
indices separately; a point's orientation can depend on several upstream angles.

Four-finger side corrections affect downstream points and tips, not their
proximal anchors. The thumb has its own side correction. Independent height
corrections apply to WRIST, the four proximal anchors, PALM, thumb metacarpal
and four placeholders. Each output point owns separate lateral and pad
corrections in `openxr_model.inc`, including zero entries. A non-thumb proximal
joint is placed with its corrections at its anchor, then carried with the MCP
flexion and oriented by the MCP link, so it is rigid on that link like the
intermediate and distal joints: the bone keeps its length and the joint turns by
the encoder angle. It therefore follows the glove's MCP link, not a measured
anatomical MCP centre. An invalid MCP angle invalidates it. Each tip is the SDK
fingertip, rigid on its distal link, oriented along the distal-to-tip direction.
Position is in meters, orientation is normalized XYZW, and all 26 output poses
are relative to WRIST. OpenXR position/orientation validity follows each point
independently; actual tracking flags require a usable wrist source.

Valid points carry separate position and orientation flags, and a partially
valid hand keeps them per point. Point radii are nominal visualization values.

## Real devices

The wrist pose comes from the OpenXR runtime through the framework's
`WristPoseSource`, as for the other glove plugins; the device page lists the
wrist sources and the aim-to-wrist options. No separate tracker process is
needed.

The plugin finds the gloves through the SDK by itself, at most one left and one
right, with nothing to declare. When both are plugged in, it opens them in one
two-hand SDK kit with time sync on, so the two hands' samples come from the same
USB frame. A glove that is plugged in alone opens on its own; when the other one
arrives, it too opens on its own first and then both are reopened together.
Gloves that cannot be opened together (no time sync, different products, ...)
keep streaming on their own and are not tried together again until either stream
ends. Until a glove is found, the plugin looks for it every 2 s and reports its
`/glove` status with the SDK's reason, without holding up the other hand. The
SDK reconnects a glove after a short dropout; when its stream ends anyway, the
plugin closes it, together with its partner when the two share a kit, and looks
for them again. A glove of the wrong product or variant, with no hand model, or
with incompatible firmware, is given up and reported as failed; an invalid SDK
configuration stops the plugin. Calibration stays in the SDK. Do not run the
plugin while the SDK viewer holds a glove's serial port.

SDK capture gates are reported as degraded on the glove output. Cached samples
do not override these statuses.

Each hand is placed with the host wrist pose at its glove sample time. Without a
tracked wrist the hand is still injected, without TRACKED flags, at the last valid
wrist pose or at the stage origin if there is none, and the `/hand` status is
degraded. Glove stalls remove the hand; the next glove sample restores it.

The plugin adds the OpenXR wrist-axis convention: WRIST is `palm_link` turned
by the reviewed WRIST rotation, with its origin moved 12.6 mm along palm_link +X,
from the back of the hand towards the palm (`kWristPadM` in `openxr_model.inc`).
That is the depth of PALM in both models; at zero flexion the middle and ring
proximal joints are within 0.5 mm of it, while the tilted thumb anchor plays no
part. Both hands use the same depth. All 26 poses and the taxel layout are
relative to this WRIST.

The `palm_link` origin is a reference point of the glove's mechanical model on
the back of the hand, near the end of the forearm, and the same point for both
models, so WRIST is a model point, not the anatomical wrist centre that OpenXR's
WRIST stands for. Nominally the wrist centre is about 4 cm further towards the
fingers and about 0.2 to 0.7 cm further towards the palm; these figures come
from typical adult hand dimensions, not from measurements on the glove, and
vary with the wearer. The plugin places the glove's WRIST at the host WRIST.
When the host wrist comes from optical hand tracking, it is the anatomical
wrist, so the injected hand sits that much further towards the fingers and the
palm than the tracked hand. With a controller, set `--aim-to-wrist-left` and
`--aim-to-wrist-right` to end at the glove's WRIST. Consumers that need the
anatomical wrist, for example to attach a forearm, must apply their own offset.
Verify the alignment before using the output for control.

## Tests

With `BUILD_TESTING=ON`, the tests build together with the plugin, so without
the C SDK they are skipped with it. Each library has its own suite under
`tests/cpp/plugins/revohuman/`:

- `core/` (`revohuman_core_tests`): the sample pipeline, tactile serialization
  and skinning, device status, event log and pose math.
- `live/` (`revohuman_live_tests`): the live source against a fake of the C SDK
  library, compiled against the SDK's header, without gloves.
- `integration/` (`revohuman_integration_tests`): both hands through the SDK
  points and the reviewed mapping, DIP-driven tips, the proximal joints rigid on
  the MCP link, each product's hand model, the P model's poses without taxels,
  the per-hand status outputs in the manifest, and the TactileMetadata built
  from the SDK taxel layout, at the bind pose and under finger bends against the
  SDK's own taxels. Two more ctest cases check how the plugin refuses the wrist
  source variable and flag.

The integration suite and the two plugin checks need the SDK library; with only
its header, the core and live suites still build. Run them all with
`ctest --test-dir build-revohuman/tests/cpp/plugins/revohuman --output-on-failure`.

## Tactile layout

At start-up the plugin builds each hand's TactileMetadata from the C SDK's
taxel layout for the DV1 model (`rh_hand_model_taxel_layout`, C SDK 0.1.3 or
later, layout version 1); there is no layout file or option. The 26 joints above
are bound where the reviewed mapping places them at zero joint angles, in the
WRIST basis. Each taxel is bound rigidly (one segment, weight 1) to the OpenXR
joint its SDK link moves with, with its +Z along the outward normal; a layout
with a taxel on a link no joint follows is refused. The log names each hand's
layout source, taxel count, metadata size and SHA-256. When the SDK has no
layout, the plugin logs why and pushes no tactile; the other outputs are
unaffected.

Reviewed XR frames are not mechanical CAD link frames, but every joint a taxel
is bound to is rigid on that taxel's SDK link. So the skinned taxels are the
SDK's at the bind pose and stay on them as the fingers bend; the integration
suite checks them against the SDK's own taxels to 1e-6 m and 1e-6 rad.

The tactile pipeline requires calibrated values and places the SDK's reading,
unchanged, on the taxel-local positive Z axis, the outward normal. The value is
the normal force the skin exerts on what it touches; the press on the skin is
its negative. Consumers use the layout to rotate into other frames. There is no
measured tangential component. Encoder and wrist validity do not gate these
local forces. Consumers need valid, time-aligned hand poses to deform the layout
or rotate forces into a wrist/world frame; bind-pose preview is only an
approximation when those poses are unavailable.

A tactile sample is pushed only when its sampling time is present, no later than
the processing clock and no more than 100 ms old, and its device time advanced
since the last pushed sample; repeated tactile ticks are suppressed. The status
names the reason a sample was not pushed: `invalid`, `uncalibrated`,
`count_mismatch`, `no_sample_time`, `stale_sample` or `nonmonotonic_sample`. An
invalid sample makes it unavailable at once, except an invalid re-send of the
last pushed sample, with the same device time, while that sample is no more than
100 ms old: it counts as a repeat. A rejected sample does not remove the forces
already pushed, so the force collection is not recreated when skipped and valid
samples alternate: the last forces keep their own sampling time and are removed
once older than 100 ms. A stream generation change, a new shape or a reset
removes them at once and clears the time anchors. Consumers must check the
tactile status as well as data presence.
