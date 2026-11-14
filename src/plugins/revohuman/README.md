<!--
SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# RevoHuman plugin

The plugin reads RevoHuman gloves through the unmodified RevoHuman C SDK and
pushes, per hand, an injected OpenXR hand, the 21 encoder joint states and the
taxel forces with their TactileMetadata. The `isaaccapture` wheel contains
neither the plugin nor the SDK; start it from the CMake install tree.

Prerequisites, running, the wrist pose and troubleshooting are on the
[RevoHuman Glove](../../../docs/source/device/hand_tracking/revohuman.rst)
device page. This README adds the build, the hand mapping and the tactile
contract.

| Directory | Contents |
| --- | --- |
| `core/` | `HandModel` (SDK hand points to OpenXR joints, TactileMetadata; the fitted keypoint offsets are in `hand_fit.inc`) and `HandPipeline` (what each hand pushes and when); unit-tested without a runtime |
| `app/` | `GloveSource` (gloves through the SDK), the OpenXR outputs, device status and `main` |

## Build and install

Follow the repository's build-from-source prerequisites first. With
`BUILD_PLUGIN_REVOHUMAN=ON`, the configure downloads the C SDK release pinned in
`CMakeLists.txt` and checks its SHA-256; `-DREVOHUMAN_SDK_ROOT=<prefix>` uses an
extracted package instead. A Python SDK wheel cannot replace the C headers and
library. From the repository root:

```bash
cmake -B build -DBUILD_PLUGIN_REVOHUMAN=ON
cmake --build build --target revohuman_plugin
cmake --install build --component revohuman
```

`<install-prefix>/plugins/revohuman` then holds the plugin, the SDK shared
library (found through `$ORIGIN`; keep its symlinks) and the SDK licence. An
incomplete SDK fails the configure.

## Gloves

The plugin finds at most one left and one right glove through the C SDK's
autokit (`rh_autokit_*`), which opens, retries and reopens them on its own
thread; `app/glove_source.cpp` turns each opened stream into hand metadata and
feeds its frames to the pipelines. Two gloves plugged in while neither is open,
as at start-up, open together in one two-hand kit, synced on one USB frame where
the SDK can (`RH_SYNC_BEST_EFFORT`). Otherwise, or when the two-hand kit cannot
open, each glove opens on its own, and a missing glove is looked for again every
`retry_ms` (the SDK default, which the plugin keeps; see `RhAutoKitOptions` in
`revohuman.h`) with the SDK reason in its `/glove` status. A glove that arrives
while the other already streams on its own also opens on its own, without
interrupting the other, and the two stay unpaired; restart the plugin with both
gloves plugged in to pair them.

The SDK reconnects a glove through dropouts and replugs itself without ending
its stream; when it reconnects one glove of a synced pair, it re-syncs the pair,
and the other glove's samples pause briefly. A stream ends only on a fatal SDK
error. A glove on its own then reopens after `retry_ms`; for a pair, both
gloves close and reopen as a pair once both are back, and if only one is back
for `pair_wait_ms` (also an SDK default) it opens on its own, the other follows on
its own, and they stay unpaired until the plugin restarts. Gloves streaming
without hardware sync to each other (host-time pairing, unpaired after a lost
pair, or opened separately) report `/glove` DEGRADED with the reason. While the
SDK's capture gate is closed a glove gives no samples, and `/glove` reports
DEGRADED with `capture gate closed (<cause>)`, or UNKNOWN before its first
sample.

A wrong product or incompatible firmware makes the SDK give the glove up, and a
glove whose hand model the plugin cannot map to OpenXR joints is not used; both
report `/glove` FAILED with the reason. The plugin keeps running, even with both
gloves given up. Do not run the plugin while the SDK viewer holds a glove's
port.

Started by hand, the plugin publishes `<id>/device_status` only with
`--plugin-root-id=<id>`; the plugin launcher passes it.

## Hand mapping

Each glove's mechanical model follows its identified product through
`rh_variant_hand_model`, and `rh_hand_model_info` gives the model's URDF
SHA-256. Every sample's angles go to `rh_hand_points`, the SDK's link poses and
fingertip points. The plugin is fitted to the PTV1 and P1 models, pinned by
URDF hash in `core/hand_fit.inc` (`kFittedModelHashes`), and fails a glove of
any other model. It places the keypoints at the OpenXR joints (wrist, joint
centres inside the fingers, tips and palm) by fixed offsets in the frames of
the SDK link poses (`kFingerFits`, `kPalmPad`, `kWristPad`, one table for
both products); the palm frame is a fixed basis in `palm_link` (`kPalmBasis`).

An SDK release that changes either pinned model's URDF moves it off the fitted
table, so refit and update `kFittedModelHashes` together with the SDK pin in
`CMakeLists.txt`. The glove's open log line names the hand model and its hash.

`core/hand_model.cpp` re-expresses the keypoints from WRIST and orients the
finger joints from the SDK link poses; WRIST and PALM keep fixed axes in
`palm_link`. The four non-thumb metacarpals are placeholders and never valid;
the four non-thumb proximal joints ride on the MCP link like the taxels bound
to it.

WRIST is a model point, not the anatomical wrist: `palm_link` moved along its
+X (towards the palm) by a depth `w`, with OpenXR X = -palm Y, Y = -palm X,
Z = -palm Z. `w` is `kWristPad` in `core/hand_fit.inc`.
The anatomical wrist centre is nominally about 4 cm towards the fingers. The
plugin puts this WRIST at the host's wrist pose, so with optical hand tracking
the injected hand sits that far off the tracked one; with a controller, set
`--aim-to-wrist-left/right` to end at the glove's WRIST. In the Tactile Metadata
Editor (`scripts/tactile_metadata`), a model rooted at `palm_link` needs URDF
Alignment Position `(0, 0, -w)`; the editor does not save it.

## Tactile

| Collection | Contract |
| --- | --- |
| `<prefix>_<side>_tactile` | `TactileDataArray`: taxel-local forces in newtons along +Z (outward normal) |
| `<prefix>_<side>_tactile/metadata` | bare `TactileMetadata`: LBS bound at zero angles, each taxel rigid on one OpenXR joint |

The metadata is built when a glove opens, from the SDK taxel layout of its model
(`rh_hand_model_taxel_layout`); a glove without tactile sensors (P1), or one
whose taxel count has no SDK layout, gets no tactile collections and a disabled
tactile status. Forces are the SDK's newtons (`rh_tactile_newtons`), unchanged.
A sample is pushed when the firmware has not marked it invalid, it is
calibrated, has the layout's taxel count, carries a sampling time no older than
`kTactileMaxAgeNs` (100 ms, in `core/inc/revohuman_plugin/pipeline.hpp`),
and both its sampling time and its device tick advanced. A skipped sample does
not remove forces already pushed; they are removed once older than that, and
the status names the skip reason. Consumers must check the tactile status as
well as data presence.

The plugin opens tactile split from the glove frames (`split_tactile` in
`RhGloveOptions`). Whether a glove splits it depends on its firmware and is
settled when the stream opens. A glove that splits it sends every tactile
sample on its own at about 215 Hz, above the 200 Hz frame rate, and the plugin
pushes each sample in sampling order; the frames' copies of the latest sample
are not used. A glove that does not split it sends tactile with the frames, and
the plugin takes one sample per frame once the stream's first `kSplitWaitFrames`
frames with tactile (`app/glove_source.hpp`) have arrived without a split
sample. The log names the source each glove uses.

### Taring the tactile sensors

Send `SIGUSR1` to the plugin (`kill -USR1 <pid>`) to tare the tactile sensors:
each streaming glove with tactile sensors gets one request and zeroes every
taxel at its current reading. This writes to the gloves, so keep both hands
still and the tactile surfaces untouched for about 3 s. The tactile stream
pauses briefly while the module tares; joint states are not affected.

Requests are never resent, and the log reports how each one ends: sent, or
otherwise a warning with the SDK's reason (for example the glove refused it, or
its result could not be confirmed). A request that cannot be made is logged with
why: the glove is not streaming, has no tactile sensors, is still running an
earlier request or is reconnecting, or its tare support is not known yet. Sent
means the glove has passed the tare to its tactile module; the glove does not
report when the module finishes it. A glove that cannot tare on request is tared
by holding its button for about 2 s. The log also reports tares started from the
button or another host, a reconnect or glove restart after which the last tare
is no longer confirmed, and a glove that reopened before its request's result
was known.

## Tests

`revohuman_core_tests` is configured with the plugin while `BUILD_TESTING` is
ON, the top-level default. It links the SDK library, not a fake, and needs
neither a glove nor an OpenXR runtime:

```bash
cmake --build build --target revohuman_core_tests
ctest --test-dir build/tests/cpp/plugins/revohuman --output-on-failure
```
