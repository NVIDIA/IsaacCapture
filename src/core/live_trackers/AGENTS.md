<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `live_trackers`

**CRITICAL (non-optional):** Before editing this package, complete the mandatory **`AGENTS.md` preflight** in [`../../../AGENTS.md`](../../../AGENTS.md) (read every applicable `AGENTS.md` on your paths, not just this file).

## Time and OpenXR

- Cache an XDev topology only after every tracker for that topology is created; a partial bind must remain retryable
  even when the enumerated IDs do not change.
- Store **`last_update_time_` as `int64_t`** (monotonic ns), not **`XrTime`**.
- **Once per `update` call:** `const XrTime xr_time = time_converter_.convert_monotonic_ns_to_xrtime(monotonic_time_ns);` then use **`xr_time`** for every **`xrLocate*`** / hand / body call **and** for MCAP (see below). **Do not** call **`convert_monotonic_ns_to_xrtime`** again in the MCAP block.
- **Full-body limp mode:** if the body tracker handle is null and you **return early**, **do not** compute **`xr_time`** first—only convert after you know you will call OpenXR.

## `DeviceDataTimestamp` (MCAP)

- **Fields 1–2:** monotonic ns (e.g. **`last_update_time_`, `last_update_time_`**).
- **Field 3 (`sample_time_raw_device_clock`):** the **same** **`xr_time`** variable used for OpenXR this frame (not a second conversion).

## Includes

- In headers that need both: **`#include <oxr_utils/oxr_funcs.hpp>`** comes **before** any bare **`#include <openxr/openxr.h>`**. `oxr_funcs.hpp` defines **`XR_NO_PROTOTYPES`** then includes OpenXR; including **`openxr.h`** first fights that policy.
- In **`.cpp`** files that construct **`DeviceDataTimestamp`**, include **`#include <schema/timestamp_generated.h>`** explicitly.
- **`.cpp`** files should include headers for **symbols the TU uses** (e.g. **`oxr_funcs.hpp`** for **`createReferenceSpace`**), not only what the matching **`.hpp`** happens to pull in.

## CMake

- **`live_trackers`** should **`PUBLIC` link `oxr::oxr_utils`** (OpenXR headers come through that INTERFACE target) because headers/sources use OpenXR / oxr types.

## Schema-based impls are generated

Live impls for trackers declared in
[`../deviceio_trackers/trackers.toml`](../deviceio_trackers/trackers.toml) are emitted into
`${CMAKE_BINARY_DIR}/generated/trackers/live_trackers/`; only hand-written trackers have `.cpp`
files in this directory. `live_deviceio_factory.{hpp,cpp}` stays hand-written but `#include`s
generated `.inc` fragments for the manifest trackers' forward decls, try-create thunks, dispatch
rows, and factory methods — **do not** add rows for a manifest tracker by hand.

Vendor routing is the reason the generated dispatch rows sit as one block at the end of
`k_tracker_dispatch`: manifest trackers are single-vendor, so their row order does not matter,
while multi-vendor hand-written types must keep their default vendor first.

## New tracker MCAP checklist

Applies to **hand-written** live tracker impls (`head`, `hand`, `controller`, `full_body`,
`message_channel`, …). For manifest trackers the impl, its MCAP
channels, and its recording traits are all generated — skip this checklist entirely.

When adding MCAP support to a new **hand-written** tracker impl, all of the following are required together—missing any one causes a build failure or wrong timestamps:

1. Add `XrTimeConverter time_converter_` and `int64_t last_update_time_ = 0` members to the impl header.
2. Initialize `time_converter_(handles)` in the constructor initializer list.
3. Declare `update(int64_t monotonic_time_ns) override` (not `XrTime`)—they are the same C++ type (`int64_t`) but semantically different; the base interface uses monotonic ns.
4. At the top of `update()`: store `last_update_time_ = monotonic_time_ns` and compute `const XrTime xr_time = time_converter_.convert_monotonic_ns_to_xrtime(monotonic_time_ns)`.
5. Use `DeviceDataTimestamp(last_update_time_, last_update_time_, xr_time)` — not `(time, time, time)`.
6. Add `MessageChannelRecordingTraits` (or equivalent) to `recording_traits.hpp` **above** its
   `generated_recording_traits.inc` include — that fragment is the manifest trackers' half.
7. **Always build** (`cmake --build <build_dir> -- -j$(nproc)`) before treating work as done. Pre-commit alone does not catch compile errors or clang-format violations enforced at build time.
8. Read `AGENTS.md` before starting. Not after CI breaks.

## Publishing tracker output

- An impl may build a `-T` as **assembly scratch** (name it `native`), but what it publishes is a `Serialized<XPayload>` encoded once per `update()`. Getters return the published handle; the scratch never escapes.
- **The scratch is a local of `update()`, never a member.** It is a temporary of one frame, so give it the lifetime of one frame — pass it to a helper by reference rather than promoting it to state. A member would outlive the encode and become a second copy of the payload that every exit path has to keep in step with the published handle; that is exactly how a tracker ends up publishing last frame's values while its scratch says otherwise. Reusing one across frames buys nothing either: the impls allocate the nested `Pose` / `HandJoints` / `ControllerPose` members fresh on every tick regardless.
- The same applies to any per-frame working buffer, not just the payload: if a member is cleared at the top of `update()` and dead by the end of it, it is a local.
- **Reset the published handle at the top of `update()`, and publish only at the bottom.** A tracker that queries the device every frame owes the caller this frame's answer or none, so invalidating on entry makes the encode the sole writer and no exit path — early return, limp mode, locate failure, or something added later by someone who never read this file — can leave last frame's snapshot readable. Prefer it to clearing on each throwing path: that only works while someone remembers to enumerate them, and a two-handed tracker has to drop *both* handles when the first hand fails, because the second was never queried.
- The exception is a tracker that **retains by design**. `SchemaTracker` leaves the last-known handle in place on a tick with no new samples and empties it only when the collection goes away, so its trackers must not reset on entry. The test is whether the tracker is query-driven (reset) or sample-driven (retain).
- Encode into a **new** buffer each frame rather than over the previous one. Consumers hold snapshots, so a caller that read last frame must keep seeing last frame's values; this is what removed the old "valid until the next `session.update()`" caveat.
- `SchemaTracker` does this for tensor-sourced trackers, and does it without encoding at all: the wire already carries the payload table, so it **adopts the sample's buffer**. Do not reintroduce an unpack on that path — the only reason it materialises a native is MCAP recording, which is why that unpack is gated on `mcap_channels_`.

## Related docs

- Manifest and generator rules: [`../codegen/AGENTS.md`](../codegen/AGENTS.md)
- Session update loop: [`../deviceio_session/AGENTS.md`](../deviceio_session/AGENTS.md)
- No OpenXR in base API: [`../deviceio_base/AGENTS.md`](../deviceio_base/AGENTS.md)
- Replay counterpart: [`../replay_trackers/AGENTS.md`](../replay_trackers/AGENTS.md)
